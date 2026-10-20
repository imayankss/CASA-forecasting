# Bank of India: public quarterly CASA figures (primary sources)

Researched 2026-10-20. Sources are Bank of India's own filings, downloaded from NSE's archive (`nsearchives.nseindia.com`, symbol BANKINDIA). BSE scrip 532149 receives the same letters. Every number below was read from a filed PDF. Where a PDF's text layer was garbled, I rendered the page and checked the number by eye.

## Summary

- **Feasible: yes.** Bank of India (BOI) files a press release and an analyst/investor presentation with the exchanges every quarter. Both give:
  - domestic CASA in ₹ crore, usually split into current account (CA) and savings account (SA);
  - the CASA ratio (%);
  - domestic deposits;
  - global deposits.
- **Coverage on NSE:**
  - **Presentations start with Q3 FY16 (Dec-2015).** The first one was filed 12-Feb-2016. NSE has no BOI presentation or press release from 2015; the 2015 results were filed only as zipped result statements ([2015 listing, NSE API](https://www.nseindia.com/api/corporate-announcements?index=equities&symbol=BANKINDIA&from_date=01-01-2015&to_date=31-12-2015)).
  - **Calendar 2015 is still recoverable.** The Mar-2016 presentation has a 5-quarter table (Mar-15, Jun-15, Sep-15, Dec-15, Mar-16) that includes Mar-2015.
  - **Latest available:** Jun-2026 (Q1 FY27), filed 24-Jul-2026.
  - **Full span:** at least Mar-2015 to Jun-2026, about 46 quarter-ends.
- **Effort:** about 40 quarterly filings to open for 2015Q1–2025Q4. That is roughly 4–6 hours by hand, or 2–3 hours with a small script plus manual checking. Details in "Effort estimate" below.
- **Easiest source by period:**
  - **2020 onwards:** the press releases are clean text, with one fixed sentence that holds all four numbers ("Domestic CASA went up by …% YoY from ₹… Cr … to ₹… Cr … and CASA ratio stood at …%").
  - **2016–2019:** data is mostly in the presentations. Filenames and categories are inconsistent, files are often zipped, and the text layer is jumbled. Plan to check visually.

## Verified figures (₹ crore; balances at quarter-end)

"Domestic deposits" and "Global deposits" are the bank's own lines. Global = domestic + overseas branches. The CASA ratio is the bank's reported figure (see the gotchas on its denominator).

| Quarter-end (BOI FY quarter) | Domestic CASA | CASA ratio (reported) | Domestic deposits | Global deposits | Source (page = PDF page) |
|---|---|---|---|---|---|
| 31-Mar-2015 (Q4 FY15) | 114,386 (CA 17,997 + SA 96,389) | 29.48% ("CASA ratio (domestic)") | 398,000 | 531,907 | Taken from the **Mar-2016** deck's comparative columns: [ANALYSTPRESENTATION (zip)](https://nsearchives.nseindia.com/corporate/ANALYSTPRESENTATION_26052016134723_222.zip) p.3 "Business - Sequential", p.4 "CASA Profile", p.2 "(29.48 % in Mar-15)". The ratio is confirmed by the [Mar-2016 press release (zip)](https://nsearchives.nseindia.com/corporate/BOIPRESSRELEASE_26052016134723_223.zip): "CASA ratio (domestic) improved from 29.48% in March 2015…" |
| 31-Mar-2019 (Q4 FY19) | 181,765 (CA 23,354 + SA 158,411) | 43.36% | 421,783 | 520,862 | [Outcome of Board Meeting, 16-May-2019](https://nsearchives.nseindia.com/corporate/BANKINDIA_16052019193043_BoiResult1_457.pdf). Press release is at p.15 and the deck is inside the same PDF: p.20 "Performance at a Glance – Business (Q-o-Q)", p.23 "CASA Profile - Domestic" |
| 31-Dec-2020 (Q3 FY21) | 214,826 (CA 29,532 + SA 185,295) | 40.61% ("share of CASA deposits") | 536,171 | 611,879 | [Press release Q3FY21](https://nsearchives.nseindia.com/corporate/BANKINDIA_10022021122346_PressReleaseDec20.pdf) p.4 table (Dec-19 / Sep-20 / Dec-20); [Q3FY21 presentation](https://nsearchives.nseindia.com/corporate/BANKINDIA_10022021123000_presentation.pdf) p.8 "Domestic Deposits - CASA" |
| 31-Mar-2023 (Q4 FY23) | 252,149 (SA 221,349 + CA 30,800) | 44.73% | 567,063 | 669,586 | [Press release Q4FY23](https://nsearchives.nseindia.com/corporate/BANKINDIA_06052023142338_PressRelease.pdf) p.3; [Q4FY23 analyst PPT](https://nsearchives.nseindia.com/corporate/BANKINDIA_06052023140826_AnalystPPT.pdf) p.6 "Performance at a Glance", p.11 "Domestic Deposit - CASA" |
| 31-Dec-2024 (Q3 FY25) | 277,190 | 41.05% | 678,625 | 794,788 | [Press release Q3FY25](https://nsearchives.nseindia.com/corporate/BANKOFINDIA_24012025170546_PressRelease.pdf) p.5; [Q3FY25 investor PPT](https://nsearchives.nseindia.com/corporate/BANKOFINDIA_24012025172000_InvestorPPT.pdf) p.8 "Domestic Deposits" (read visually because the slide is an image); [provisional figures 03-Jan-2025](https://nsearchives.nseindia.com/corporate/BANKOFINDIA_03012025164932_Q3FY25results.pdf) give 678,626 domestic and 794,787 global |

### More quarters read from the same documents' comparison columns (verified)

| Quarter-end | Domestic CASA | CASA ratio | Domestic dep. | Global dep. | Source |
|---|---|---|---|---|---|
| 30-Jun-2015 | 115,737 | not given | 376,360 | 508,260 | Mar-2016 deck p.3–4 (link above) |
| 30-Sep-2015 | 119,513 (CA 18,723 + SA 100,790) | not given | 389,864 | 525,195 | Mar-2016 deck p.3–4 |
| 31-Dec-2015 | 123,907 | not given | 391,480 | 528,772 | Mar-2016 deck p.3–4 |
| 31-Mar-2016 | 127,924 (CA 19,443 + SA 108,481) | 34.18% | 377,309 | 513,005 | Mar-2016 deck p.2–4 |
| 31-Mar-2018 | 172,787 (CA 25,579 + SA 147,208) | 41.43% | 421,211 | 520,854 | Mar-2019 PDF p.20, p.23 |
| 30-Jun-2018 | 169,916 (CA 24,733 + SA 145,183) | 41.76% | 410,442 | 514,604 | Mar-2019 PDF p.20, p.23 |
| 30-Sep-2018 | 170,205 (CA 20,526 + SA 149,679) | 41.44% | 413,921 | 512,142 | Mar-2019 PDF p.20, p.23 |
| 31-Dec-2018 | 175,010 (CA 23,881 + SA 151,129) | 42.65% | 413,449 | 515,128 | Mar-2019 PDF p.20, p.23 |
| 31-Dec-2019 | 191,735 (CA 28,110 + SA 163,625) | 42.69% | 453,458 | 522,138 | Dec-2020 press release p.4 and deck p.8 |
| 31-Mar-2020 | 197,751 (CA 26,046 + SA 171,704) | not given | not given | not given | Dec-2020 deck p.8 |
| 30-Sep-2020 | 207,281 in the press release; 207,280 in the deck | 39.49% | 532,095 | 607,529 | Dec-2020 press release p.3–4 |
| 31-Mar-2022 | 245,464 (SA 215,638 + CA 29,826) | 45.02% | 550,833 | not checked | Mar-2023 PPT p.11; press release p.3 |
| 31-Dec-2022 | 246,718 (SA 214,048 + CA 32,670) | 44.56% | 556,707 | not checked | Mar-2023 PPT p.10–11 |
| 31-Dec-2023 | 261,335 | 43.88% | 599,137 | 707,827 | Q3FY25 press release p.5 and PPT p.8 |
| 30-Sep-2024 | 275,618 | 41.18% | not checked | not checked | Q3FY25 PPT p.8 |
| 30-Jun-2025 | 281,846 | not checked | 710,277 | 833,698 | [Q1FY27 press release](https://nsearchives.nseindia.com/corporate/BANKINDIA6_24072026170444_pressrelease30062026.pdf) p.4 |
| 30-Jun-2026 (latest) | 302,085 | 36.68% | 824,963 | 957,924 | Q1FY27 press release p.4 (link above) |

## Where the documents are

**NSE archive (reliable, worked with plain `curl`):**
- File URLs follow `https://nsearchives.nseindia.com/corporate/<SYMBOL-ish>_<DDMMYYYYHHMMSS>_<free-text>.pdf`. The prefix varies: `BANKINDIA_`, `BANKOFINDIA_`, `BANKINDIA2_`, `BANKINDIA5_`, `BANKINDIA6_`. The free-text part is arbitrary, for example `PressReleaseDec20`, `AnalystPPT`, `InvestorPPT`, `pressrelease30062026`. **You cannot guess the URLs; enumerate them.**
- Filings up to about 2018 are `.zip` files with one PDF inside, for example `ANALYSTPRESENTATION_26052016134723_222.zip`.
- **To enumerate,** use the NSE JSON endpoint: `https://www.nseindia.com/api/corporate-announcements?index=equities&symbol=BANKINDIA&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY`. It needs a browser User-Agent and the cookies from first loading an nseindia.com page. Each record has `desc` (for example "Investor Presentation", "Press Release", "Outcome of Board Meeting", "Analysts/Institutional Investor Meet/Con. Call Updates"), `attchmntText` and `attchmntFile`. It worked here for every year from 2015 to 2026.
- **Filing categories by period** (from those listings):
  - **2016–2019:** the deck is filed under "Analysts Meet", "Updates", "Analysts/Institutional Investor Meet/Con. Call Updates" or inside "Outcome of Board Meeting". Example filenames: `BOIPre_12022016…zip`, `BOIPresentationDecember2016_…zip`, `BOIPresentations_10082017…zip`, `BOIPressPresentation_10112017…zip`, `BOIpress_29052018…zip`, `BOIPRPRE_01082018…zip`, `BANKINDIA_13112018…_boipress_539.pdf`, `BANKINDIA_28012019…_boipresentation_316.pdf`, and Mar-2019 inside `…_BoiResult1_457.pdf`.
  - **2020 onwards:** separate "Investor Presentation" and "Press Release" filings every quarter: Jan/Jun/Aug/Nov 2020, then every results date up to 24-Jul-2026.
  - **Since about 2024:** a "Provisional Business Figures" filing about 3 days after each quarter-end gives global and domestic deposits **but not CASA**. Example: [03-Jan-2025](https://nsearchives.nseindia.com/corporate/BANKOFINDIA_03012025164932_Q3FY25results.pdf).
- **Download gotcha:** large NSE PDFs were cut off mid-transfer twice; the Q3FY25 deck arrived as 5.5 MB of 10.2 MB. Compare the file size with `Content-Length` and resume (`curl -C -`).

**BSE:**
- Attachments are at `https://www.bseindia.com/xml-data/corpfiling/AttachHis/<uuid>.pdf`, and a direct `curl` of one worked.
- The listing API (`api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/...`) returned 403 "Access Denied" to scripts. That makes BSE hard to enumerate, and NSE is the better index.
- WebFetch also got 403 on the BSE PDF.

**Bank of India website:** see "Unverified" below. Both `bankofindia.co.in/investor-relations` and the new domain `bankofindia.bank.in/...` returned HTTP 403 with a Cloudflare "Just a moment..." challenge page. I did not try to get past it.

## Text-based or scanned?

Text was extracted with macOS PDFKit.

| Document | Text layer |
|---|---|
| Mar-2016 deck and press release (zip) | **Text.** Tables come out flattened and out of order, so check visually. |
| Mar-2019 board-outcome PDF (56 pp.) | **Text** for the press release and deck pages. The cover letter is a scan with a poor OCR layer. |
| Dec-2020 press release and deck | **Text.** The press release table extracts cleanly. |
| Mar-2023 press release and deck | **Text.** The cover letter is a scan with OCR; the body is clean. |
| May-2024 (Q4FY24) deck | **Text.** The deposit table extracts. |
| **Jan-2025 (Q3FY25) deck** | **Image-based.** Only slide titles extract; the tables are pictures, so you need OCR or reading by eye. The press release from the same date **is text**. |
| Jul-2026 (Q1FY27) press release | **Text.** |

So for any quarter, the press release is the safest machine-readable source.

## FY convention and calendar mapping

- The BOI financial year runs from 1 April to 31 March. FY25 = FY 2024-25. Every document I checked uses this:
  - "Q3FY25 … Quarter/9-Months ended 31st Dec. 2024" ([Q3FY25 press release](https://nsearchives.nseindia.com/corporate/BANKOFINDIA_24012025170546_PressRelease.pdf));
  - "Q3FY21 Results" for Dec-2020;
  - "Q4'FY23 & FY 2022-23" for Mar-2023;
  - "MARCH 2019 (Q4 & FY 2018-19)";
  - "Q4/Full Year : FY15-16" for Mar-2016.
- **Quarter mapping:**

| BOI quarter | Months | Calendar quarter |
|---|---|---|
| Q1 | Apr–Jun | Q2 |
| Q2 | Jul–Sep | Q3 |
| Q3 | Oct–Dec | Q4 |
| Q4 | Jan–Mar | Q1 of the FY's end year |

- **For your series:**
  - 2015Q1 = 31-Mar-2015 = Q4 FY15.
  - 2019Q1 = Q4 FY19.
  - 2020Q4 = Q3 FY21.
  - 2023Q1 = Q4 FY23.
  - 2024Q4 = Q3 FY25.
  - 2025Q4 = Q3 FY26 (results filed 21-Jan-2026).
- All the figures are **quarter-end balances (stocks)**, not quarterly averages or flows.

## Gotchas

1. **The reported CASA ratio is not CASA ÷ the "Domestic deposits" headline.** For example:

   | Quarter | CASA ÷ domestic deposits | Reported ratio |
   |---|---|---|
   | Dec-24 | 277,190 / 678,625 = 40.85% | 41.05% |
   | Mar-23 | 44.47% | 44.73% |
   | Dec-20 | 40.07% | 40.61% |
   | Mar-19 | 43.09% | 43.36% |
   | Mar-15 | 28.74% | 29.48% |

   The denominator implied by the reported ratio is ₹2.6k–10k crore smaller than headline domestic deposits.
   - It is not consistently CASA + term deposits either. That formula matches exactly for Dec-24 and Dec-23 (CASA + "Total Term Deposits", 41.05% and 43.88%), but not for Sep-24 (gives 41.01% vs 41.18% reported) or Mar-23 (gives 44.47% vs 44.73%), using the [Q3FY25 deck](https://nsearchives.nseindia.com/corporate/BANKOFINDIA_24012025172000_InvestorPPT.pdf) p.8 and the [Q4FY24 deck](https://nsearchives.nseindia.com/corporate/BANKINDIA_10052024175008_InvestorPPT.pdf).
   - **Recommendation:** store the reported ratio as its own column, and compute your own CASA ÷ domestic deposits for consistency.
2. **"Total Deposits" in a "Performance at a Glance" table means global deposits.** Example: Mar-23 "Total Deposits 6,69,586" ([Q4FY23 PPT](https://nsearchives.nseindia.com/corporate/BANKINDIA_06052023140826_AnalystPPT.pdf) p.6). The CASA amount and ratio are always domestic.
3. **Small differences between documents.**
   - Provisional vs final: Dec-24 domestic 678,626 vs 678,625; global 794,787 vs 794,788.
   - Press release vs deck: Sep-20 CASA 207,281 vs 207,280.
   - Comparative columns may be restated in later decks. Prefer the quarter's own document, and use comparatives only to fill gaps.
4. **The comparison columns change from deck to deck:**
   - Mar-2016 and Mar-2019 decks: 5 quarters.
   - Dec-2020 deck: Dec-19 / Mar-20 / Sep-20 / Dec-20. **Jun-20 is skipped.**
   - Mar-2023 deck: Mar-22 / Dec-22 / Mar-23.
   - Q3FY25 deck: Dec-23 / Sep-24 / Dec-24.
   - The Mar-2016 deck gives CASA ratios for Mar-15 and Mar-16 only, not Jun/Sep/Dec-15.
5. **Wording changes:** "CASA ratio (domestic)" (2016), "The ratio of CASA" (2019), "share of CASA deposits" (2020), "CASA percentage" (2023), "CASA ratio" (2025). Write the regex to allow for all of these.
6. **The CASA level shifted sharply:** 29.48% in Mar-15, about 45% in Mar-22, 36.68% in Jun-26. That partly reflects BOI shedding bulk deposits in FY16 (press release: "shedding Bulk Deposits to Rs.58,980 Crores", [Mar-2016 press release](https://nsearchives.nseindia.com/corporate/BOIPRESSRELEASE_26052016134723_223.zip)) and a later shift towards bulk deposits. A real series will have structural breaks that the synthetic one probably lacks.
7. **Search engines mislabel bank decks.** A web-search summary credited a "CASA base @50.58%" deck to BOI. I downloaded it ([BSE AttachHis 775c0797…](https://www.bseindia.com/xml-data/corpfiling/AttachHis/775c0797-d7d4-4d2a-a9c9-fd805796dbb7.pdf)) and it is **Central Bank of India**'s Mar-2022 deck (scrip CENTRALBK / 532885). Check the issuer on page 1 of every PDF.

## Effort estimate (2015Q1–2025Q4 = 44 quarter-ends)

- **Number of documents:**
  - 2015Q1–2015Q4: 1 document (the Mar-2016 deck's comparatives). CASA ratio for Jun/Sep/Dec-15 needs the Dec-2015 deck ([`BOIPre_12022016162154_369.zip`](https://nsearchives.nseindia.com/corporate/BOIPre_12022016162154_369.zip), not opened) or must be left blank.
  - 2016Q1–2019Q4: about 16 documents.
  - 2020Q1–2025Q4: about 24 press releases.
  - **Total: about 40 documents.** Using the multi-quarter comparison tables could cut this to about 20–25, at some risk of restated values.
- **Layout consistency:**
  - 2020+: high in the press releases. One sentence pattern, plus a table in 2020–21.
  - 2016–2019: medium to low. Zips, varying filing categories, jumbled text order.
  - 2024+ decks may be images, but the press releases cover the same numbers.
- **Time:**
  - About 2–3 minutes per quarter for 2020+ (about 1–1.5 h).
  - About 8–10 minutes per quarter for 2016–2019 (about 2.5 h).
  - Plus cross-checks: **about 4–6 hours by hand.**
  - A script (NSE API listing → download with resume → PDFKit/pdfplumber text → regex on press releases) plus a visual check of the 2016–2019 decks: **about 2–3 hours.**
- **Aggregators (not investigated in depth; another agent covers RBI):** I found no source that publishes BOI's *quarterly* CASA in a more convenient form. RBI's bank-wise data and BOI's annual-report "Schedule 3 – Deposits" (demand / savings / term, India vs outside India) are presumably **annual** (March) only. Unverified.

## Unverified

- **BOI website pages and URL patterns.** Everything on `bankofindia.co.in` and `bankofindia.bank.in` returned a Cloudflare challenge (HTTP 403, "Just a moment...") to WebFetch and curl. A web search lists a page `https://bankofindia.bank.in/investor-presentation-sebi` with analyst-meet notices for 2022–2024, but I could not open it. A Nov-2023 BSE filing reportedly says presentations are "also being made available on our Bank's website", but I did not open that filing. Whether the website holds decks from before 2016, including an original Mar-2015 deck, is unverified.
- **An original Mar-2015 (Q4 FY15) presentation or press release.** NSE has none. BSE may have one but could not be enumerated. The Mar-2015 figures above come from the Mar-2016 deck's comparative columns and the Mar-2016 press release.
- **The exact CASA-ratio denominator.** No definition was found in the documents I read. Whether it excludes interbank deposits, CDs or something else is unverified.
- **Completeness for 2016–2019, quarter by quarter.** NSE listings show deck or press-release filings for Dec-15, Mar-16, Sep-16 (press release only), Dec-16, Jun-17, Sep-17, Dec-17, Mar-18, Jun-18, Sep-18, Dec-18, Mar-19, Jun-19 and Sep-19. I only opened Mar-16 and Mar-19. For Jun-16 and Mar-17 I saw no deck-named filing; they may sit in result zips or "Analyst" filings with other names, or be recoverable from the next deck's comparatives.
- **Dec-2025 (Q3 FY26) figures in secondary media** (CASA ratio 37.97%) were not checked against the filing ([`BANKINDIA5_21012026164706_PressReleaseQ3.pdf`](https://nsearchives.nseindia.com/corporate/BANKINDIA5_21012026164706_PressReleaseQ3.pdf)).
- **RBI or other aggregators** giving BOI's quarterly CASA: not investigated beyond the note above.
