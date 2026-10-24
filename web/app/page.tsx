import { BoiExplorer } from "@/components/results/BoiExplorer";
import { CapacityFrontier } from "@/components/results/CapacityFrontier";
import { OriginsChart } from "@/components/results/OriginsChart";
import { ThemeToggle } from "@/components/results/ThemeToggle";
import { readJson } from "@/lib/data";
import type { GainsRow } from "@/lib/frontier";

type Row = Record<string, number | string>;
type LeadModel = { lift_at_10pct: number; lift_at_10pct_ci95: [number, number]; roc_auc: number; pr_auc: number };
type Boi = { series: Row[]; leaderboard: { model: string; MAPE: number }[]; transfer_mape: { model: string; MAPE_pct: number }[]; forecasts: Row[] };
type Panel = {
  panel_rows: number;
  banks: number;
  origins: string[];
  leaderboard: { model: string; MASE_mean: number; MASE_median: number; series_origins: number }[];
  paired_bootstrap_vs_AutoETS: { a: string; mean_diff: number; ci95: [number, number] }[];
  stress_2023Q1: Record<string, { MASE_h1_2023Q1: number; MASE_h1_other: number }>;
  aggregate_ape_pct_h1: Record<string, number>;
  mase_by_origin: Row[];
  run: { git_sha: string | null };
};
type Leads = {
  test_customers: number;
  new_adds_only: { customers: number; adds: number; base_rate: number; share_of_adds_excluded: number; models: Record<string, LeadModel> };
  gains_new_adds: GainsRow[];
};

const REPO = "https://github.com/imayankss/CASA-forecasting";
// Santander dataset size (README, Track A); the result files only hold the scored test month.
const SANTANDER = { customers: 956_645, rows: "13.6M" };
const PANEL_LABEL: Record<string, string> = {
  Naive: "No change (baseline)", AutoTheta: "AutoTheta", AutoETS: "AutoETS (per bank)",
  Chronos2: "Chronos-2 zero-shot", GlobalLGBM: "Global LightGBM", SeasonalNaive: "Seasonal naive (baseline)"
};
const LEAD_LABEL: Record<string, string> = {
  lightgbm: "LightGBM", logistic_regression: "Logistic regression", rule_active_x_products: "CRM rule (active × products)"
};

function Section({ n, id, title, children }: { n: string; id: string; title: string; children: React.ReactNode }) {
  return (
    <section className="section" id={id}>
      <h2><span className="num">{n}</span>{title}</h2>
      {children}
    </section>
  );
}

export default async function ResultsPage() {
  const [boi, panel, leads] = await Promise.all([
    readJson<Boi | null>("boi_real.json", null),
    readJson<Panel | null>("panel.json", null),
    readJson<Leads | null>("propensity.json", null)
  ]);
  if (!boi || !panel || !leads) {
    return <main className="page"><p className="hint">Result files missing. Run <code>python scripts/export_web_data.py</code>.</p></main>;
  }

  const mape = Object.fromEntries(boi.transfer_mape.map((r) => [r.model, r.MAPE_pct]));
  const bestFitted = boi.leaderboard.filter((r) => !["Drift", "SeasonalNaive"].includes(r.model)).sort((a, b) => a.MAPE - b.MAPE)[0];
  const quarters = boi.series.length;
  const holdout = boi.forecasts.length;

  const board = [...panel.leaderboard].sort((a, b) => a.MASE_mean - b.MASE_mean);
  const naive = board.find((r) => r.model === "Naive")!;
  const ets = board.find((r) => r.model === "AutoETS")!;
  const lgbmVsEts = panel.paired_bootstrap_vs_AutoETS.find((r) => r.a === "GlobalLGBM")!;
  const lgbmStress = panel.stress_2023Q1.GlobalLGBM;

  const na = leads.new_adds_only;
  const lgb = na.models.lightgbm;
  const sha = panel.run.git_sha?.slice(0, 7) ?? "uncommitted";

  return (
    <main className="page">
      <header>
        <div className="masthead">
          <h1 className="title">CASA Deposit Forecasting</h1>
          <ThemeToggle />
        </div>
        <p className="claim">
          Bank deposit forecasting and CASA lead scoring rebuilt on real public data at three scales — every model measured against a simple baseline, on periods it never saw.
        </p>
        <dl className="meta">
          <div><dt>Origin</dt><dd>Bank of India data science internship</dd></div>
          <div><dt>Data</dt><dd>Public: NSE filings · FDIC · Kaggle</dd></div>
          <div><dt>Evaluation</dt><dd>Later periods only, baselines always</dd></div>
          <div><dt>Scale</dt><dd>{quarters} quarters · {panel.banks.toLocaleString()} banks · {(SANTANDER.customers / 1000).toFixed(0)}k customers</dd></div>
          <div><dt>Panel run</dt><dd><a href={`${REPO}/commit/${panel.run.git_sha}`}>{sha}</a></dd></div>
          <div><dt>Code · Licence</dt><dd><a href={REPO}>GitHub</a> · MIT</dd></div>
        </dl>
      </header>

      <div className="lede">
        <p>
          A deposit forecast is easy to make look good. On Bank of India&apos;s real CASA series, a straight line drawn through the history misses by just <strong>{mape.Drift.toFixed(2)}%</strong> over {holdout} quarters — better than SARIMA, Prophet and AutoARIMA. A model that doesn&apos;t report that baseline is hiding it.
        </p>
        <p>
          This project began as an internship forecaster run on synthetic data. It now runs on three real public datasets, and every table below puts the models next to the baseline they have to beat.
        </p>
        <div className="cells">
          <div className="cell">
            <span className="label">Bank of India</span>
            <span className="value baseline">{mape.Drift.toFixed(2)}% MAPE</span>
            <span className="note">drift baseline wins; best fitted model {bestFitted.MAPE.toFixed(2)}%</span>
          </div>
          <div className="cell">
            <span className="label">{panel.banks.toLocaleString()} US banks</span>
            <span className="value baseline">{naive.MASE_mean.toFixed(2)} MASE</span>
            <span className="note">&ldquo;no change&rdquo; beat every model (AutoETS {ets.MASE_mean.toFixed(2)})</span>
          </div>
          <div className="cell">
            <span className="label">{(SANTANDER.customers / 1000).toFixed(0)}k customers</span>
            <span className="value model">{lgb.lift_at_10pct.toFixed(2)}× lift</span>
            <span className="note">top 10% vs random, 95% CI {lgb.lift_at_10pct_ci95[0].toFixed(2)}–{lgb.lift_at_10pct_ci95[1].toFixed(2)}</span>
          </div>
        </div>
      </div>

      <Section n="01" id="scales" title="Three scales, three questions">
        <ol className="steps">
          <li><strong>One bank, {quarters} quarters.</strong> Bank of India&apos;s CASA, read from its own quarterly filings on NSE with every figure cited to a page. How well can it be forecast?</li>
          <li><strong>{panel.banks.toLocaleString()} banks, {panel.panel_rows.toLocaleString()} bank-quarters.</strong> FDIC Call Reports, 2001–2026, with transaction plus savings deposits as the US analogue of CASA. Do global models beat per-bank models across thousands of banks?</li>
          <li><strong>{SANTANDER.customers.toLocaleString()} customers, {SANTANDER.rows} customer-months.</strong> Kaggle&apos;s Santander dataset, which contains no real customers. Which customers will open a CASA-type account next month?</li>
        </ol>
      </Section>

      <Section n="02" id="protocol" title="Evaluation protocol">
        <div className="prose"><p>The rules were fixed before any result was read. Each one closes a way a forecast can look better than it is.</p></div>
        <ol className="steps">
          <li><strong>Later periods only.</strong> No random splits. Every metric is measured on quarters or months after the training data ends.</li>
          <li><strong>Leakage guards that raise.</strong> <code>assert_temporal_split</code> and <code>assert_features_known_at</code> stop the run if training data or a feature reaches into the test period.</li>
          <li><strong>A baseline in every table.</strong> Drift, seasonal naive, &ldquo;no change&rdquo;, and a rule a CRM team could write without a model.</li>
          <li><strong>An uncertainty range where the sample allows.</strong> Bootstrap intervals over banks and over customers.</li>
          <li><strong>No survivorship bias.</strong> Failed banks such as SVB and Signature stay in the panel.</li>
          <li><strong>Provenance on every run.</strong> Each result file records its config, the sha256 of its inputs and the git commit.</li>
        </ol>
      </Section>

      <Section n="03" id="boi" title="Bank of India: the real CASA series">
        <div className="prose">
          <p>Pick a model to see its forecast for the {holdout}-quarter holdout against what Bank of India actually reported. The LightGBM model never saw BOI: it was trained only on US banks and applied to BOI&apos;s growth pattern.</p>
        </div>
        <BoiExplorer history={boi.series} forecasts={boi.forecasts} mape={mape} />
        <div className="callout">
          The model trained on {panel.banks.toLocaleString()} US banks ({mape.GlobalLGBM_FDIC.toFixed(2)}%) <strong>beat every model fitted on BOI itself</strong> — except the straight line ({mape.Drift.toFixed(2)}%).
        </div>
      </Section>

      <Section n="04" id="panel" title={`${panel.banks.toLocaleString()} US banks: when “no change” wins`}>
        <div className="prose">
          <p>
            Forecasts from {panel.origins.length} origins, {panel.origins[0].slice(0, 7)} to {panel.origins.at(-1)!.slice(0, 7)}, 1–4 quarters ahead. Trend-following models learned from two decades of mostly rising deposits, then forecast growth just as the 2022–23 rate hikes reversed it.
          </p>
        </div>
        <OriginsChart rows={panel.mase_by_origin} />
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Model</th><th className="num">MASE mean</th><th className="num">MASE median</th><th className="num">1-step, 2023Q1</th><th className="num">Sum-of-banks error</th></tr>
            </thead>
            <tbody>
              {board.map((r, i) => (
                <tr key={r.model} className={i === 0 ? "best" : undefined}>
                  <td>{PANEL_LABEL[r.model] ?? r.model}</td>
                  <td className="num">{r.MASE_mean.toFixed(2)}</td>
                  <td className="num">{r.MASE_median.toFixed(2)}</td>
                  <td className="num">{panel.stress_2023Q1[r.model]?.MASE_h1_2023Q1.toFixed(2)}</td>
                  <td className="num">{panel.aggregate_ape_pct_h1[r.model]?.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="callout">
          <strong>The global model ties per-bank AutoETS:</strong> mean MASE difference {lgbmVsEts.mean_diff >= 0 ? "+" : ""}{lgbmVsEts.mean_diff.toFixed(2)}, 95% CI {lgbmVsEts.ci95[0].toFixed(3)} to {lgbmVsEts.ci95[1].toFixed(3)}. In the 2023Q1 outflow quarter its one-step error rose from {lgbmStress.MASE_h1_other.toFixed(2)} to {lgbmStress.MASE_h1_2023Q1.toFixed(2)}.
        </div>
      </Section>

      <Section n="05" id="leads" title="Lead scoring: the call-capacity frontier">
        <div className="prose">
          <p>
            A score is half a decision. The other half is how many customers a branch team can call. Slide the capacity and see how many of the {na.adds.toLocaleString()} genuinely new CASA openers that month the model&apos;s ranking reaches. Customers re-opening an account they held in the last five months ({(na.share_of_adds_excluded * 100).toFixed(0)}% of all opens) are excluded: they are easy to predict and worth little.
          </p>
        </div>
        <CapacityFrontier gains={leads.gains_new_adds} />
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Model</th><th className="num">Lift @ top 10%</th><th className="num">95% CI</th><th className="num">ROC AUC</th></tr>
            </thead>
            <tbody>
              {Object.entries(na.models).sort((a, b) => b[1].lift_at_10pct - a[1].lift_at_10pct).map(([k, m], i) => (
                <tr key={k} className={i === 0 ? "best" : undefined}>
                  <td>{LEAD_LABEL[k] ?? k}</td>
                  <td className="num">{m.lift_at_10pct.toFixed(2)}×</td>
                  <td className="num">{m.lift_at_10pct_ci95[0].toFixed(2)}–{m.lift_at_10pct_ci95[1].toFixed(2)}</td>
                  <td className="num">{m.roc_auc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="prose"><p>Trained on June 2015 to February 2016, early-stopped on March, tested on April features with May labels.</p></div>
      </Section>

      <Section n="06" id="fixed" title="What I fixed in the original project">
        <ol className="steps">
          <li><strong>SARIMAX used future data.</strong> It forecast with the actual repo rate, CPI and GDP of the test quarters. Macro inputs are now lagged by the full horizon, and a guard enforces it.</li>
          <li><strong>ARIMA forecast a flat line.</strong> With no drift term it could not represent growth. Added the trend term.</li>
          <li><strong>The drift monitor could not fail.</strong> It compared levels of a trending series, so it always alarmed. It now compares quarter-on-quarter growth.</li>
          <li><strong>No baselines, no significance tests.</strong> Added seasonal-naive and drift baselines, MASE, and Diebold–Mariano tests.</li>
        </ol>
      </Section>

      <Section n="07" id="reproduce" title="Reproduction">
        <div className="prose"><p>Every number on this page is read from a committed result file. To regenerate them:</p></div>
        <pre>{`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-scale.txt
.venv/bin/python run_pipeline.py --data data/real/boi_casa_quarterly.csv --out-dir data/processed/boi_real
.venv/bin/python run_panel_ingest.py && .venv/bin/python run_panel.py
.venv/bin/python run_propensity.py      # needs the Kaggle data in data/external/santander
.venv/bin/python scripts/export_web_data.py`}</pre>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Data</th><th>Source</th><th>Status</th></tr></thead>
            <tbody>
              <tr><td>BOI quarterly CASA</td><td>Bank of India filings on NSE, each row cited</td><td>Real, public, in git</td></tr>
              <tr><td>FDIC Call Reports</td><td>api.fdic.gov</td><td>Real, public, downloaded by script</td></tr>
              <tr><td>Santander Product Recommendation</td><td>Kaggle competition</td><td>Public sample, no real customers; not redistributed</td></tr>
              <tr><td>Original 40-quarter series</td><td>LLM-generated</td><td>Synthetic; test input only</td></tr>
            </tbody>
          </table>
        </div>
      </Section>

      <footer className="footer">
        CASA Deposit Forecasting · results read from reports/ and data/processed/ · <a href={REPO}>source</a>
      </footer>
    </main>
  );
}
