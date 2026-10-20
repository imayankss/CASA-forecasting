"""
santander.py — Santander Product Recommendation data for CASA-acquisition scoring.

Kaggle's data page states the sample contains no real Santander Spain
customers. The raw data may not be redistributed, so it lives only under
data/external/ (gitignored).
"""

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "external" / "santander"
TRAIN_CSV = RAW_DIR / "train_ver2.csv"
TRAIN_PARQUET = RAW_DIR / "train.parquet"

PRODUCTS = [
    "ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1", "ind_cder_fin_ult1",
    "ind_cno_fin_ult1", "ind_ctju_fin_ult1", "ind_ctma_fin_ult1", "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1", "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1",
    "ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1",
    "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1",
]

# Our reading of "CASA-type" accounts; Kaggle does not classify the products.
# Current, particular / particular plus / mas particular, e-account, payroll
# account, savings. Junior accounts (ind_ctju) are excluded.
CASA_PRODUCTS = [
    "ind_cco_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1", "ind_ctma_fin_ult1",
    "ind_ecue_fin_ult1", "ind_cno_fin_ult1", "ind_ahor_fin_ult1",
]

CATEGORICAL = [
    "ind_empleado", "pais_residencia", "sexo", "indrel_1mes", "tiprel_1mes", "indresi",
    "indext", "canal_entrada", "indfall", "nomprov", "segmento",
]


def ingest(csv_path: Path = TRAIN_CSV, out_path: Path = TRAIN_PARQUET) -> Path:
    """Stream the raw CSV into a typed Parquet file without loading it into memory.

    The CSV has padded numbers (" 35", "     NA") and mixed codes ("1", "1.0"),
    so every column is read as text and cast leniently.
    """
    lf = pl.scan_csv(csv_path, infer_schema=False)
    strip = lambda c: pl.col(c).str.strip_chars()
    lf = lf.with_columns(
        pl.col("fecha_dato").str.to_date("%Y-%m-%d"),
        pl.col("fecha_alta").str.to_date("%Y-%m-%d", strict=False),
        strip("ncodpers").cast(pl.Int32),
        strip("age").cast(pl.Int16, strict=False),
        strip("antiguedad").cast(pl.Int16, strict=False),
        strip("ind_nuevo").cast(pl.Int8, strict=False),
        strip("indrel").cast(pl.Int8, strict=False),
        strip("cod_prov").cast(pl.Int8, strict=False),
        strip("ind_actividad_cliente").cast(pl.Int8, strict=False),
        strip("renta").cast(pl.Float32, strict=False),
        # "1.0" and "1" are the same code
        strip("indrel_1mes").str.replace(r"\.0$", ""),
        *[strip(c).cast(pl.Int8, strict=False) for c in PRODUCTS],
    ).with_columns(
        *[pl.col(c).cast(pl.Categorical) for c in CATEGORICAL],
    ).drop("ult_fec_cli_1t", "conyuemp", "tipodom")
    lf.sink_parquet(out_path)
    return out_path


def load(path: Path = TRAIN_PARQUET) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def _month_number() -> pl.Expr:
    return (pl.col("fecha_dato").dt.year().cast(pl.Int32) * 12
            + pl.col("fecha_dato").dt.month().cast(pl.Int32)).alias("mnum")


FEATURES_PARQUET = RAW_DIR / "features.parquet"
DEMOGRAPHICS = [
    "age", "antiguedad", "renta", "ind_actividad_cliente", "ind_nuevo", "indrel", "cod_prov",
    "segmento", "sexo", "canal_entrada", "tiprel_1mes", "indrel_1mes", "ind_empleado",
    "indext", "indresi",
]
CASA_LAGS = range(2, 6)


def build_features(lf: pl.LazyFrame, first_month: str = "2015-06-28",
                   out_path: Path = FEATURES_PARQUET) -> Path:
    """Feature table dated at month t, using only data from t and earlier.

    Every lag is a join on mnum + k with k >= 1, so no feature can come from
    after t. Joined with casa_targets, so each row carries its t+1 label.
    """
    flags = lf.select("ncodpers", _month_number(),
                      *[pl.col(c).fill_null(0).cast(pl.Int8) for c in PRODUCTS])

    def lagged(k: int, cols):
        return flags.select("ncodpers", (pl.col("mnum") + k).alias("mnum"),
                            *[pl.col(c).alias(f"{c}_l{k}") for c in cols])

    base = (
        lf.filter(pl.col("fecha_dato") >= pl.lit(first_month).str.to_date())
        .select("ncodpers", "fecha_dato", _month_number(), *DEMOGRAPHICS,
                *[pl.col(c).fill_null(0).cast(pl.Int8) for c in PRODUCTS])
        .with_columns(
            pl.col("renta").is_null().cast(pl.Int8).alias("renta_missing"),
            ((pl.col("fecha_dato").dt.month() % 12) + 1).cast(pl.Int8).alias("label_month"),
        )
    )
    feats = base.join(lagged(1, PRODUCTS), on=["ncodpers", "mnum"], how="left")
    for k in CASA_LAGS:
        feats = feats.join(lagged(k, CASA_PRODUCTS), on=["ncodpers", "mnum"], how="left")

    n_now = pl.sum_horizontal(PRODUCTS)
    n_prev = pl.sum_horizontal([pl.col(f"{c}_l1").fill_null(0) for c in PRODUCTS])
    adds_last = pl.sum_horizontal(
        [((pl.col(c) == 1) & (pl.col(f"{c}_l1").fill_null(0) == 0)).cast(pl.Int8) for c in PRODUCTS])
    feats = feats.with_columns(
        n_now.cast(pl.Int8).alias("n_products"),
        (n_now - n_prev).cast(pl.Int8).alias("n_products_change"),
        adds_last.cast(pl.Int8).alias("n_added_last_month"),
        pl.col("ind_cco_fin_ult1_l1").is_null().cast(pl.Int8).alias("new_customer"),
    )
    targets = casa_targets(lf).select("ncodpers", "fecha_dato", "casa_add")
    feats.join(targets, on=["ncodpers", "fecha_dato"], how="inner").drop("mnum").sink_parquet(out_path)
    return out_path


def casa_targets(lf: pl.LazyFrame) -> pl.LazyFrame:
    """casa_add = 1 if the customer holds a CASA product at t+1 they did not hold at t.

    Rows are dated at the feature month t. Only customers present at both t and
    t+1 are scored. Missing flags count as "not held". This is "adds", never
    "holds": ownership barely changes month to month, so a "holds" target would
    inflate lift.
    """
    own = lf.select(
        "ncodpers", "fecha_dato", _month_number(),
        *[pl.col(c).fill_null(0).cast(pl.Int8) for c in CASA_PRODUCTS],
    )
    nxt = own.select(
        "ncodpers", (pl.col("mnum") - 1).alias("mnum"),
        *[pl.col(c).alias(f"{c}_next") for c in CASA_PRODUCTS],
    )
    added = pl.any_horizontal(
        [(pl.col(f"{c}_next") == 1) & (pl.col(c) == 0) for c in CASA_PRODUCTS]
    )
    return (
        own.join(nxt, on=["ncodpers", "mnum"], how="inner")
        .select("ncodpers", "fecha_dato", added.cast(pl.Int8).alias("casa_add"))
    )
