"""Real quarterly CASA figures for Bank of India (NSE symbol BANKINDIA).

Pipeline:
1. ``list_announcements`` pulls BOI's corporate announcements from NSE's JSON
   API (one call per calendar year, cached as JSON).
2. ``select_filings`` keeps the press releases and analyst/investor
   presentations, which carry the CASA figures.
3. ``download`` fetches each attachment from nsearchives.nseindia.com into
   ``data/external/boi_filings/`` (gitignored), resuming truncated transfers
   and unpacking the pre-2019 ``.zip`` filings.
4. ``extract_*`` read the figures with pdfplumber. Press releases from
   2019 onwards state CASA in one sentence; the tables in older decks come out
   jumbled, so those quarters (and a few image-only pages) are read by eye and
   kept in ``MANUAL`` below with their source URL and page.
5. ``build_rows`` merges everything into one row per quarter-end and
   ``write_csv`` writes ``data/real/boi_casa_quarterly.csv``.

Amounts are in Rs crore exactly as the bank reports them. Nothing is
computed: a value not stated in a source is left blank, and the CASA ratio is
the bank's own reported figure (its denominator is not domestic deposits; see
docs/research/boi-public-casa.md).

BOI's financial year runs April to March, so Q3FY25 is the quarter ending
31-Dec-2024.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "external" / "boi_filings"
OUT_CSV = ROOT / "data" / "real" / "boi_casa_quarterly.csv"

NSE_HOME = "https://www.nseindia.com/get-quotes/equity?symbol=BANKINDIA"
NSE_API = (
    "https://www.nseindia.com/api/corporate-announcements"
    "?index=equities&symbol=BANKINDIA&from_date=01-01-{y}&to_date=31-12-{y}"
)
ARCHIVE = "https://nsearchives.nseindia.com/corporate/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

COLUMNS = [
    "quarter_end",
    "fy_quarter",
    "casa_cr",
    "ca_cr",
    "sa_cr",
    "casa_ratio_reported",
    "domestic_deposits_cr",
    "global_deposits_cr",
    "source_url",
    "source_page",
]
FIGURES = ["casa_cr", "ca_cr", "sa_cr", "casa_ratio_reported", "domestic_deposits_cr", "global_deposits_cr"]


# --------------------------------------------------------------------------
# Quarter helpers
# --------------------------------------------------------------------------
def quarter_ends(start: dt.date = dt.date(2015, 3, 31), end: dt.date = dt.date(2026, 6, 30)) -> list[dt.date]:
    out, y, q = [], start.year, (start.month - 1) // 3
    while True:
        m = 3 * q + 3
        d = dt.date(y, m, 31 if m in (3, 12) else 30)
        if d > end:
            return out
        if d >= start:
            out.append(d)
        q += 1
        if q == 4:
            q, y = 0, y + 1


def fy_quarter(d: dt.date) -> str:
    """BOI label: FY runs Apr-Mar, so Dec-2024 is Q3FY25 and Mar-2025 is Q4FY25."""
    fy = d.year + 1 if d.month > 3 else d.year
    q = {6: 1, 9: 2, 12: 3, 3: 4}[d.month]
    return f"Q{q}FY{fy % 100:02d}"


def last_quarter_end_before(d: dt.date) -> dt.date:
    qe = [q for q in quarter_ends(dt.date(2014, 12, 31), dt.date(2027, 3, 31)) if q < d]
    return qe[-1]


# --------------------------------------------------------------------------
# NSE enumeration and download
# --------------------------------------------------------------------------
class NSEClient:
    def __init__(self, cache_dir: Path = CACHE_DIR, pause: float = 1.5):
        self.cache_dir = cache_dir
        self.pause = pause
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": NSE_HOME,
            }
        )
        self._primed = False

    def _get(self, url: str, **kw) -> requests.Response:
        delay = self.pause
        for attempt in range(5):
            try:
                r = self.s.get(url, timeout=120, **kw)
                if r.status_code in (200, 206, 416):
                    time.sleep(self.pause)
                    return r
                log.warning("HTTP %s for %s", r.status_code, url)
            except requests.RequestException as e:  # pragma: no cover - network
                log.warning("%s for %s", e, url)
            time.sleep(delay)
            delay *= 2
        raise RuntimeError(f"giving up on {url}")

    def list_announcements(self, years=range(2015, 2027)) -> list[dict]:
        path = self.cache_dir / "announcements.json"
        if path.exists():
            return json.loads(path.read_text())
        if not self._primed:  # NSE API needs the cookies set by a normal page load
            self._get(NSE_HOME)
            self._primed = True
        rows: list[dict] = []
        for y in years:
            rows += self._get(NSE_API.format(y=y)).json()
        path.write_text(json.dumps(rows))
        return rows

    def download(self, url: str) -> Path:
        """Download with resume and a Content-Length check; unzip .zip filings."""
        name = url.rsplit("/", 1)[-1]
        dest = self.cache_dir / name
        if _complete(dest):
            return unzip_pdf(dest) if dest.suffix.lower() == ".zip" else dest
        for _ in range(6):
            have = dest.stat().st_size if dest.exists() else 0
            head = self.s.head(url, timeout=60, allow_redirects=True)
            total = int(head.headers.get("Content-Length", 0) or 0)
            if have and total and have >= total:
                break
            headers = {"Range": f"bytes={have}-"} if have else {}
            r = self._get(url, headers=headers, stream=True)
            mode = "ab" if (have and r.status_code == 206) else "wb"
            with dest.open(mode) as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
            if not total:
                break
        if dest.suffix.lower() == ".zip":
            return unzip_pdf(dest)
        return dest


def _complete(path: Path) -> bool:
    """A cached file is reused only if it is a whole zip / PDF (NSE transfers do get cut off)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.suffix.lower() == ".zip":
        return zipfile.is_zipfile(path)
    with path.open("rb") as fh:
        fh.seek(max(0, path.stat().st_size - 2048))
        return b"%%EOF" in fh.read()


def unzip_pdf(zpath: Path) -> Path:
    out = zpath.with_suffix(".pdf")
    if not out.exists():
        with zipfile.ZipFile(zpath) as z:
            pdfs = [n for n in z.namelist() if n.lower().endswith(".pdf")]
            if not pdfs:
                raise RuntimeError(f"no PDF in {zpath}")
            out.write_bytes(z.read(pdfs[0]))
    return out


@dataclass
class Filing:
    date: dt.date
    desc: str
    text: str
    url: str

    @property
    def kind(self) -> str:
        blob = f"{self.desc} {self.text} {self.url}".lower()
        if "press release" in self.desc.lower() or re.search(r"press_?release|presrelease|boipress|prs\.pdf|press\.pdf|pressrekease", blob):
            return "press"
        return "deck"


DECK_PAT = re.compile(r"presentation|ppt|analys|perfanalysis|boipre|prpre|ilovepdf", re.I)
EXCLUDE_PAT = re.compile(
    r"transcript|intimation|schedule|meet with|one to one|one-to-one|audio|recording|"
    r"credit rating|care|icra|newspaper|interaction|makrana|bofa|investormeet|analystmeet|"
    r"invite|invitation|concall|call|trasncript|egm|iepf|analystme\.pdf|groupmeet|analyst1?\d\d\.pdf|"
    # one-page "schedule of analyst meet" letters: Analyst_<ts>.zip, BOIanalyst_<ts>.zip, ..._boianalyst_<n>.pdf
    r"(?:^|_)(?:boi|nsebse)?analyst\d*(?:_\d+)*\.(?:pdf|zip)$",
    re.I,
)


def select_filings(ann: list[dict]) -> list[Filing]:
    out = []
    for a in ann:
        url = a.get("attchmntFile") or ""
        if not url:
            continue
        desc, text = a.get("desc", ""), a.get("attchmntText") or ""
        blob = f"{desc} {text} {url}"
        is_press = desc.lower().startswith("press release")
        is_deck = desc in ("Investor Presentation", "Analysts Meet") or (
            "Analysts/Institutional" in desc and DECK_PAT.search(url.rsplit("/", 1)[-1])
        ) or (desc == "Updates" and re.search(r"presentation", text, re.I))
        if not (is_press or is_deck):
            continue
        if EXCLUDE_PAT.search(url.rsplit("/", 1)[-1]) or re.search(r"credit rating", blob, re.I):
            continue
        d = dt.datetime.strptime(a["an_dt"][:11], "%d-%b-%Y").date()
        out.append(Filing(d, desc, text, url))
    out.sort(key=lambda f: (f.date, f.url))
    return out


# Filings that carry CASA but are not labelled as a press release or deck.
EXTRA_FILINGS = [
    # Mar-2019 press release and deck are inside the board-meeting outcome PDF.
    Filing(dt.date(2019, 5, 16), "Outcome of Board Meeting", "", ARCHIVE + "BANKINDIA_16052019193043_BoiResult1_457.pdf"),
    # Jun-2019 press release was filed under "Analysts/Institutional Investor Meet".
    Filing(dt.date(2019, 7, 30), "Press Release", "", ARCHIVE + "BANKINDIA_30072019153512_pressrelease_193.pdf"),
]


def assign_quarters(filings: list[Filing]) -> dict[dt.date, list[Filing]]:
    """Map each filing to the quarter it reports: the last quarter-end before it was filed."""
    out: dict[dt.date, list[Filing]] = {}
    for f in filings:
        out.setdefault(last_quarter_end_before(f.date), []).append(f)
    return out


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------
def page_texts(pdf_path: Path) -> list[tuple[int, str]]:
    """(viewer page number, text) per page, via pdfplumber, cached as JSON.

    pdfplumber silently drops some pages (the scanned cover letters of several
    2017-2020 filings), which would shift every page number after them. Page
    numbers are therefore taken from pypdfium2, which counts pages the way a
    PDF viewer does, by aligning the two on page text.
    """
    cache = pdf_path.with_suffix(".pages.json")
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    import pdfplumber

    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            try:
                texts.append(pg.dedupe_chars().extract_text() or "")
            except Exception:  # pragma: no cover - malformed page
                texts.append("")
    out = list(zip(viewer_page_numbers(pdf_path, texts), texts))
    cache.write_text(json.dumps(out))
    return out


def _tokens(t: str) -> set[str]:
    return set(re.findall(r"[A-Za-z]{3,}|\d[\d,.]{2,}", t))


def viewer_page_numbers(pdf_path: Path, texts: list[str]) -> list[int]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    n = len(doc)
    if n == len(texts):
        return list(range(1, n + 1))
    viewer = [_tokens(doc[k].get_textpage().get_text_range()) for k in range(n)]
    out, j, slack = [], 0, n - len(texts)
    for t in texts:
        mine = _tokens(t)
        cands = range(j, min(n, j + max(slack, 0) + 1))
        best = max(cands, key=lambda k: len(mine & viewer[k]) / (len(mine | viewer[k]) or 1))
        out.append(best + 1)
        slack -= best - j
        j = best + 1
    return out


# Amounts: Indian grouping (2,77,190), western (277,190) or bare (277190), optional decimals.
NUM = r"(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d{4,7}(?:\.\d+)?)"
PCT = r"(\d{2}\.\d{1,2}|\d{2})\s?%"
RS = r"(?:Rs\.?\s?|₹\s?|`\.?\s?)"


def num(s: str) -> float:
    v = float(s.replace(",", ""))
    return int(v) if v.is_integer() else v


@dataclass
class Hit:
    value: float
    page: int


def _last_amount(line: str) -> float | None:
    nums = re.findall(NUM, line)
    return num(nums[-1]) if nums else None


def _flat(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("’", "'"))


# Sentence patterns (searched on whitespace-flattened page text). The current
# quarter's value is the one after "to"/"stood at"; the "from" value is the
# year-ago comparative and is ignored.
SENTENCE_PATTERNS: dict[str, list[re.Pattern]] = {
    "casa_cr": [
        re.compile(rf"Domestic CASA (?:deposits )?went up by [\d.]+ ?% Y[oO]Y (?:from {RS}{NUM} (?:Cr|crore)(?: in [A-Za-z'.]+,?(?: ?'?\d{{2,4}})?)? )?to {RS}{NUM} (?:Cr|crore)"),
        re.compile(rf"CASA level (?:has )?(?:increased|improved) from {RS}{NUM} crore in [A-Za-z]+,? ?\d{{4}} to {RS}{NUM} crore"),
        re.compile(rf"CASA Deposits \({RS}{NUM} crore\)"),
    ],
    "casa_ratio_reported": [
        re.compile(rf"Domestic CASA (?:deposits )?went up by .{{0,120}}?CASA (?:ratio|percentage) (?:stood at|improved from [\d.]+% in \S+ to) {PCT}"),
        re.compile(rf"(?:The )?CASA ratio stood at {PCT}"),
        re.compile(rf"(?:ratio of CASA|CASA ratio \(domestic(?: operations)?\)|share in [Dd]omestic deposits|share of CASA deposits)(?: has)? (?:improved|stood) (?:sequentially )?(?:qoq )?(?:from {PCT} in [A-Za-z.]+,? ?\d{{4}} )?(?:to|at) {PCT}"),
        re.compile(rf"CASA (?:ratio|share) (?:stands |improves to |improved to )?(?:at )?{PCT}"),
    ],
    "global_deposits_cr": [
        re.compile(rf"Global Deposits (?:increased|went up)(?: by)?(?: [\d.]+ ?%)?(?: Y[oO]Y)?(?: and stood at| from {RS}{NUM} (?:Cr|crore)(?: in [A-Za-z'.]+,?(?: ?'?\d{{2,4}})?)? to| to) {RS}{NUM}"),
    ],
    "domestic_deposits_cr": [
        re.compile(rf"Domestic [Dd]eposits stood at {RS}{NUM}"),
        re.compile(rf"Domestic [Dd]eposits (?:increased|went up)(?: by)?(?: [\d.]+ ?%)?(?: Y[oO]Y)?(?: and stood at| from {RS}{NUM} (?:Cr|crore)(?: in [A-Za-z'.]+,?(?: ?'?\d{{2,4}})?)? to| to) {RS}{NUM}"),
    ],
    "sa_cr": [re.compile(rf"Savings deposits increased by [\d.]+% YoY to {RS}{NUM}")],
    "ca_cr": [re.compile(rf"Current deposits increased by [\d.]+% YoY to {RS}{NUM}")],
}

# Table rows (searched line by line). Columns run oldest -> newest, so the
# current quarter is the last amount on the line.
ROW_PATTERNS: dict[str, list[re.Pattern]] = {
    "casa_cr": [
        re.compile(rf"^Domestic CASA {NUM}"),
        re.compile(rf"^CASA \(Amount(?: / %)?\) {NUM}"),
        re.compile(rf"^CASA {NUM} {NUM}"),
        re.compile(rf"^CASA (?:Deposits|deposits)(?:\(Domestic\))? {NUM} -?[\d.]+%?"),
    ],
    "sa_cr": [re.compile(rf"^Savings? (?:Account|Deposits?) {NUM}(?: {NUM})+")],
    "ca_cr": [re.compile(rf"^Current (?:Account|Deposits?) {NUM}(?: {NUM})+")],
    "global_deposits_cr": [re.compile(rf"^(?:Global Deposits|Deposits \(Global\)) {NUM}")],
    "domestic_deposits_cr": [re.compile(rf"^Domestic deposits {NUM}")],
}


def _row_last_pct(line: str) -> float | None:
    """Last percentage on a table row; whole tokens only, so dates like 31.03.25 are skipped."""
    toks = line.replace("/", " ").split()
    pcts = [m.group(1) for t in toks if (m := re.fullmatch(r"(\d{2}\.\d{1,2})%?", t))]
    return float(pcts[-1]) if pcts else None


def extract_document(pages: list[tuple[int, str]]) -> dict[str, Hit]:
    """Pull the current quarter's figures out of one press release / deck."""
    found: dict[str, Hit] = {}

    def put(field_: str, value, page: int) -> None:
        if value is not None and field_ not in found:
            found[field_] = Hit(value, page)

    for i, text in pages:
        flat = _flat(text)
        for field_, pats in SENTENCE_PATTERNS.items():
            for pat in pats:
                m = pat.search(flat)
                if m:
                    g = m.groups()[-1]
                    put(field_, float(g) if field_ == "casa_ratio_reported" else num(g), i)
                    break
        lines = [ln.strip() for ln in text.replace("’", "'").splitlines()]
        for j, ln in enumerate(lines):
            for field_, pats in ROW_PATTERNS.items():
                if any(p.search(ln) for p in pats):
                    put(field_, _last_amount(ln), i)
            # Domestic deposits: first "Domestic <amounts>" row after the global-deposits row.
            if re.match(r"^(?:Global Deposits|Deposits \(Global\)) ", ln):
                for nxt in lines[j + 1 : j + 3]:
                    if re.match(rf"^Domestic {NUM}", nxt):
                        put("domestic_deposits_cr", _last_amount(nxt), i)
                        break
            # Row label on its own line ("Current" / "Account" wrapped around the numbers).
            if re.fullmatch(r"(Current|Savings?)(?: Account)?", ln) and j + 1 < len(lines):
                if re.fullmatch(rf"{NUM}(?: {NUM})+(?: -?\(?[\d.]+\)?%?)?", lines[j + 1]):
                    put("ca_cr" if ln.startswith("Current") else "sa_cr", _last_amount(lines[j + 1]), i)
            if re.match(r"^CASA Ratio \(%\) ", ln) or re.match(r"^CASA \(%\) ", ln):
                put("casa_ratio_reported", _row_last_pct(ln), i)
            if re.match(r"^CASA \(Amount / %\) ", ln):
                put("casa_ratio_reported", _row_last_pct(ln), i)
            m = re.findall(r"CASA ?(?:Ratio)? ?: ?(\d{2}\.\d{2}) ?%", ln)
            if m:
                put("casa_ratio_reported", float(m[-1]), i)
    return found


PLAUSIBLE = {
    "casa_cr": (100_000, 400_000),
    "ca_cr": (10_000, 50_000),
    "sa_cr": (80_000, 300_000),
    "casa_ratio_reported": (25, 50),
    "domestic_deposits_cr": (300_000, 900_000),
    "global_deposits_cr": (450_000, 1_100_000),
}


def plausible(field_: str, value: float) -> bool:
    lo, hi = PLAUSIBLE[field_]
    return lo <= value <= hi


# --------------------------------------------------------------------------
# Row assembly
# --------------------------------------------------------------------------
@dataclass
class Row:
    quarter_end: dt.date
    values: dict[str, float] = field(default_factory=dict)
    # source url -> {page -> [fields]}
    sources: dict[str, dict[int, list[str]]] = field(default_factory=dict)

    def set(self, field_: str, value, url: str, page) -> None:
        self.values[field_] = value
        self.sources.setdefault(url, {}).setdefault(page, []).append(field_)

    def csv_row(self) -> dict[str, str]:
        out = {"quarter_end": self.quarter_end.isoformat(), "fy_quarter": fy_quarter(self.quarter_end)}
        for f in FIGURES:
            v = self.values.get(f)
            out[f] = "" if v is None else (f"{v:.2f}" if f == "casa_ratio_reported" else str(v))
        urls = list(self.sources)
        out["source_url"] = " | ".join(urls)
        out["source_page"] = " | ".join(
            "; ".join(f"p.{pg} {','.join(fs)}" for pg, fs in sorted(self.sources[u].items()))
            for u in urls
        )
        return out


def _ordered(q: dt.date, filings: list[Filing]) -> list[Filing]:
    """Press releases first from Jun-2019 (clean text); decks first before that (more precise)."""
    press_first = q >= dt.date(2019, 6, 30)
    return sorted(filings, key=lambda f: ((f.kind != "press") if press_first else (f.kind == "press"), f.date))


def build_rows(client: NSEClient | None = None) -> list[Row]:
    client = client or NSEClient()
    filings = select_filings(client.list_announcements()) + EXTRA_FILINGS
    by_q = assign_quarters(sorted(filings, key=lambda f: f.date))
    rows = []
    for q in quarter_ends():
        row = Row(q)
        manual = MANUAL.get(q.isoformat(), {})
        for f in _ordered(q, by_q.get(q, [])):
            hits = extract_document(page_texts(client.download(f.url)))
            for field_, hit in hits.items():
                if field_ in row.values or field_ in manual or not plausible(field_, hit.value):
                    continue
                row.set(field_, hit.value, f.url, hit.page)
        for field_, (value, url, page) in manual.items():
            if field_ == "note":
                continue
            row.set(field_, value, url, page)
        _check_consistency(row)
        if row.values:
            rows.append(row)
    return rows


def _check_consistency(row: Row) -> None:
    v = row.values
    if {"ca_cr", "sa_cr", "casa_cr"} <= v.keys() and abs(v["ca_cr"] + v["sa_cr"] - v["casa_cr"]) > 2:
        log.warning("%s: CA + SA != CASA (%s + %s vs %s)", row.quarter_end, v["ca_cr"], v["sa_cr"], v["casa_cr"])
    if {"domestic_deposits_cr", "global_deposits_cr"} <= v.keys() and v["domestic_deposits_cr"] >= v["global_deposits_cr"]:
        log.warning("%s: domestic >= global deposits", row.quarter_end)
    if {"domestic_deposits_cr", "casa_cr"} <= v.keys() and v["casa_cr"] >= v["domestic_deposits_cr"]:
        log.warning("%s: CASA >= domestic deposits", row.quarter_end)


def write_csv(rows: list[Row], path: Path = OUT_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r.quarter_end):
            w.writerow(r.csv_row())
    return path


# --------------------------------------------------------------------------
# Values read by eye. Used where the quarter has no filing of its own on NSE
# (taken from a later filing's comparative columns), or where the text layer is
# an image or a chart whose labels come out jumbled. Each entry is
# field -> (value, source_url, page). These override anything extracted.
# --------------------------------------------------------------------------
_A = ARCHIVE
DEC15_DECK = _A + "BOIPre_12022016162154_369.zip"  # Q3FY16 deck: 5-quarter tables Dec-14..Dec-15
MAR16_DECK = _A + "ANALYSTPRESENTATION_26052016134723_222.zip"
MAR16_PR = _A + "BOIPRESSRELEASE_26052016134723_223.zip"
SEP16_PR = _A + "BOIPress_10112016144205_215.zip"
DEC16_DECK = _A + "BOIPresentationDecember2016_13022017163627_400.zip"
JUN17_DECK = _A + "BOIPresentations_10082017183730_335.zip"  # tables Jun-16..Jun-17
SEP17_DECK = _A + "BOIPressPresentation_10112017171757_344.zip"
DEC17_DECK = _A + "BOIPress_12022018194730_324.zip"
MAR18_DECK = _A + "BOIpress_29052018132222_150.zip"
JUN18_DECK = _A + "BOIPRPRE_01082018150437_264.zip"
SEP18_DECK = _A + "BANKINDIA_13112018134039_boipress_539.pdf"
DEC18_DECK = _A + "BANKINDIA_28012019171503_boipresentation_316.pdf"
MAR19_DECK = _A + "BANKINDIA_16052019193043_BoiResult1_457.pdf"
MAR22_DECK = _A + "BANKINDIA_24052022131535_SEBOIPerfAnalysisQ4.pdf"

# Pages are viewer page numbers (pdfium count, including scanned cover letters).
# "CASA Profile" slides before 2019 are charts: their text layer scatters the
# labels, so CA and SA were read from the rendered slide.
MANUAL: dict[str, dict[str, tuple]] = {
    # No Mar-2015 filing on NSE: comparative columns of the Dec-2015 deck.
    "2015-03-31": {
        "casa_cr": (114386, DEC15_DECK, 4), "ca_cr": (17997, DEC15_DECK, 4), "sa_cr": (96389, DEC15_DECK, 4),
        "casa_ratio_reported": (29.48, MAR16_PR, 1),  # "CASA ratio (domestic) improved from 29.48% in March 2015"
        "domestic_deposits_cr": (398000, DEC15_DECK, 3), "global_deposits_cr": (531907, DEC15_DECK, 3),
    },
    "2015-06-30": {
        "casa_cr": (115737, DEC15_DECK, 4), "ca_cr": (19376, DEC15_DECK, 4), "sa_cr": (96361, DEC15_DECK, 4),
        "domestic_deposits_cr": (376360, DEC15_DECK, 3), "global_deposits_cr": (508260, DEC15_DECK, 3),
    },
    "2015-09-30": {
        "casa_cr": (119513, DEC15_DECK, 4), "ca_cr": (18723, DEC15_DECK, 4), "sa_cr": (100790, DEC15_DECK, 4),
        "casa_ratio_reported": (31.0, SEP16_PR, 1),  # rounded: "improved from 31% in September 2015"
        "domestic_deposits_cr": (389864, DEC15_DECK, 3), "global_deposits_cr": (525195, DEC15_DECK, 3),
    },
    "2015-12-31": {"ca_cr": (22555, DEC15_DECK, 4), "sa_cr": (101352, DEC15_DECK, 4)},
    "2016-03-31": {"ca_cr": (19443, MAR16_DECK, 4), "sa_cr": (108481, MAR16_DECK, 4)},
    # No Jun-2016 deck or press release on NSE: comparatives in the Jun-2017 deck.
    "2016-06-30": {
        "casa_cr": (131272, JUN17_DECK, 9), "ca_cr": (21196, JUN17_DECK, 9), "sa_cr": (110077, JUN17_DECK, 9),
        "domestic_deposits_cr": (373290, JUN17_DECK, 8), "global_deposits_cr": (498000, JUN17_DECK, 8),
    },
    # Sep-2016 has only a press release (ratio "36%"); amounts from the Jun-2017 deck.
    "2016-09-30": {
        "casa_cr": (134789, JUN17_DECK, 9), "ca_cr": (19473, JUN17_DECK, 9), "sa_cr": (115316, JUN17_DECK, 9),
        "domestic_deposits_cr": (381724, JUN17_DECK, 8), "global_deposits_cr": (505280, JUN17_DECK, 8),
    },
    "2016-12-31": {"ca_cr": (27104, DEC16_DECK, 4), "sa_cr": (143281, DEC16_DECK, 4)},
    # No Mar-2017 deck on NSE (only the meeting notice): Jun-2017 deck and Mar-2018 press release.
    "2017-03-31": {
        "casa_cr": (166608, JUN17_DECK, 9), "ca_cr": (23498, JUN17_DECK, 9), "sa_cr": (143111, JUN17_DECK, 9),
        "casa_ratio_reported": (39.84, MAR18_DECK, 2),  # "The ratio of CASA has improved from 39.84% in March, 2017"
        "domestic_deposits_cr": (423457, JUN17_DECK, 8), "global_deposits_cr": (540032, JUN17_DECK, 8),
    },
    "2017-06-30": {"ca_cr": (24364, JUN17_DECK, 9), "sa_cr": (139968, JUN17_DECK, 9)},
    "2017-09-30": {"ca_cr": (21101, SEP17_DECK, 9), "sa_cr": (141669, SEP17_DECK, 9)},
    "2017-12-31": {"ca_cr": (24041, DEC17_DECK, 12), "sa_cr": (141579, DEC17_DECK, 12)},
    "2018-03-31": {"ca_cr": (25579, MAR18_DECK, 10), "sa_cr": (147208, MAR18_DECK, 10)},
    "2018-06-30": {"ca_cr": (24733, JUN18_DECK, 11), "sa_cr": (145183, JUN18_DECK, 11)},
    # Own deck shows CASA 170,206 (p.7, p.9); later decks restate Sep-18 as 170,205.
    "2018-09-30": {"ca_cr": (20526, SEP18_DECK, 9), "sa_cr": (149679, SEP18_DECK, 9)},
    "2018-12-31": {
        "casa_cr": (175010, DEC18_DECK, 6), "ca_cr": (23881, DEC18_DECK, 6), "sa_cr": (151129, DEC18_DECK, 6),
    },
    "2019-03-31": {"ca_cr": (23354, MAR19_DECK, 23), "sa_cr": (158411, MAR19_DECK, 23)},
    # Q4FY22 press release gives no domestic deposits; deck chart "Domestic and Overseas Deposits".
    "2022-03-31": {"domestic_deposits_cr": (550833, MAR22_DECK, 8)},
}


def _formats(v: float) -> set[str]:
    """Ways a figure can be printed: 1,77,190 / 177,190 / 177190 / 1,70,205."""
    if not float(v).is_integer():
        return {f"{v}", f"{v:,}"}
    n = int(v)
    s = str(n)
    indian = s[-3:]
    rest = s[:-3]
    while rest:
        indian = rest[-2:] + "," + indian
        rest = rest[:-2]
    return {s, f"{n:,}", indian, f"{n}.00", f"{v:.2f}"}


def verify_manual(client: NSEClient) -> list[str]:
    """Report manual values whose number does not appear in the cited page's text layer."""
    problems = []
    for q, fields in MANUAL.items():
        for f, (v, url, page) in fields.items():
            pages = dict(page_texts(client.download(url)))
            text = pages.get(page, "")
            if not any(x in text for x in _formats(v)):
                problems.append(f"{q} {f}={v} not in text of {url.rsplit('/', 1)[-1]} p.{page}")
    return problems
