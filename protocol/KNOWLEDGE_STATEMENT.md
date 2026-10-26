# Knowledge statement

## 1. What I know about electricity market conditions after 30 June 2026

| Item | A. Nothing | B. General awareness only (e.g. news headlines, weather) | C. Specific (prices, spreads, reports) |
|---|---|---|---|
| Day-ahead or real-time prices, or DART spreads, for any of the four markets | ☒ | ☐ | ☐ |
| Market-monitor or ISO reports covering July–December 2026 | ☒ | ☐ | ☐ |
| News about price spikes, scarcity or extreme weather in these markets, July–December 2026 | ☒ | ☐ | ☐ |
| Anything else that could suggest how DART spreads behaved after June 2026 | ☒ | ☐ | ☐ |

Details for any B or C: none

## 2. Facts stated by the analysis team (no action needed; listed so the record is complete)

1. **2026-H1 data was used before the earlier freeze (FORENSICS §5a).** The protocol behind the published version was hashed before the 33-node 2026-H1 panel was pulled. But 2026-H1 prices for three nodes had already been used on 21 July 2026, in an earlier calendar-only cross-market holdout. The 2026-H1 results are therefore treated as weaker evidence than a sealed test.
2. **Documents dated after June 2026 that research agents opened or cited (FORENSICS §5c; audit record `docs/fee_rederivation_2026-10-23.md`, sections 8 and 9).** The keys in parentheses (I4, N5 and so on) are source keys of the fee note, `docs/virtual_bidding_rules_and_costs.md`, and of the audit record.

   On 22 October 2026, during research on virtual-bidding fees, the fee note drew on seven documents dated, or now served, after 30 June 2026. The audit record finds that the note's fee levels relied on six of them (N5, I7, M4, M6, C5 and I1 as cited). The seventh, I4, was not used for any fee. For each document, the entry gives what was taken from it and what replaced it when the calibrated fees were re-derived on 23 October 2026 from sources dated no later than 30 June 2026 (Rulings 78 and 90 in the analysis log):
   - ISO-NE's October 2026 virtual-bid financial-assurance proxy file (I4). Taken: a research agent accessed the file and reported DA–RT proxies of about $14–22/MWh, which FORENSICS §5c calls a single coarse figure. The analysis log refers to the fee note's "$14-22 and $17/MWh lines", and the record does not say whether $17/MWh is a separate figure, so the figures seen are stated as about $14–22/MWh and $17/MWh. The proxies are computed from recent DA–RT differences, which probably include summer 2026, and summer 2026 overlaps Test A's window. FORENSICS §5c and the audit record report no other figure about market conditions after June 2026. Replaced by: nothing. No fee uses the file, and the lines were redacted from the fee note.
   - NYISO training file `6-Ancillary-Services.pdf` (N5), created 6 October 2026. Taken: the 2026 Rate Schedule 1 rate for virtual transactions, $0.1919 per cleared MWh. Replaced by: NYISO's posting of the 2024 non-physical Schedule 1 rates (N4, posted 3 August 2023), which gives $0.1333 per cleared MWh.
   - ISO-NE Tariff Section IV.A (I7), file created 28 August 2026. Taken: the Schedule 2 charges of $0.005 per increment offer or decrement bid submitted and $0.06 per one that clears. Replaced by: the same values from ISO-NE's filing of 15 October 2024 on recovering its 2025 administrative costs (A2).
   - The iso-ne.com copy of the 2025 market-monitor report (I1), modified 15 July 2026. The report's cover says June 2026, but this copy was probably posted after June, so it is listed out of caution. Taken: the 2025 uplift charge rates, $0.63/MWh for ISO-NE, below $0.1/MWh for NYISO and $0.16/MWh for MISO. Replaced by: the same values from the Potomac Economics copy of the report (A1, created 11 June 2026).
   - MISO's energy-market rates notice for October 2026 (M4). Taken: the Schedule 17 administrative rate, $0.0746/MWh, which the fee note carried. Replaced by: nothing. No compliant source was found for MISO's Schedule 17 and Schedule 24 rates, so both are omitted from the calibrated MISO fees, and the omission is disclosed.
   - MISO's 2025 State of the Market report, posted July 2026 (M6). It describes 2025. Taken: the 2025 real-time RSG rate for MISO, $0.16/MWh, used in the MISO INC fee. Replaced by: the same $0.16/MWh from Table 2 of A1.
   - CAISO tariff Section 11 as of 22 July 2026 (C5). It states settlement rules and has no price information. Taken: the settlement and bid-cost-recovery rules. Replaced by: CAISO tariff Section 11 as of 19 November 2025 (A5).

   CAISO's Department of Market Monitoring 2025 annual report (C8) is not on this list. The audit record verifies that it was created on 25 June 2026 (PDF metadata; running header "June 2026"), so it is compliant, and it is used. The audit record lists three figures taken from it: a 2025 bid-cost-recovery figure of $0.42 per cleared virtual-supply MWh, a restated 2024 figure of $0.35 and a 55% clearing figure. Only the 2025 figure enters the calibrated fees, as part of the CAISO INC fee. FORENSICS §5c first listed C8 as probably dated after June; its correction note withdraws that.

   For fee levels, each of the seven sources above was replaced by a compliant source or omitted, so no document dated after 30 June 2026 sets a calibrated fee level.

   During the re-derivation, on 23 October 2026, a research agent opened eleven files whose dates turned out to be after 30 June 2026. Ten were current versions of tariff sections, manuals or training files. The eleventh was I1, the iso-ne.com copy of the 2025 market-monitor report. They were:
   - three listed above: I1, N5 and I7;
   - seven further sources that the fee note had cited for rules rather than fee levels: NYISO's consolidated tariff (N1) and training file `7-Virtual-Trading.pdf` (N2); ISO-NE's Tariff Section I (I2), Market Rule 1 Appendix F (I5) and Section I Exhibit IA (I8); CAISO's tariff Section 30 (C2); and MISO's BPM-005 manual package (M3);
   - one candidate substitute for C5, which was not used: the May 2026 version of CAISO's tariff Section 11, created 13 July 2026.

   For each file the agent read only the dates (the file metadata, the effective-date stamps of the NYISO tariff, or the entry list of the MISO package; for I1 also page 1), used and recorded no number from it, and deleted its local copy. The rules taken from I5 and M3 were re-sourced from compliant documents. The fee note's eligibility, credit and limit lines that cite N1, N2, I2, I4, I8 and C2, and two non-fee context lines that cite M6 and C5, were not re-sourced, and the audit found no compliant substitute for them. None of them sets a calibrated fee level.

   Search results also displayed titles and short summaries of other documents dated after June 2026, which were not opened. The summaries covered MISO's October 2026 rates notice (its Schedule 17, 24 and 16 rates), ISO-NE filings of July to October 2026 (2027 Schedule 2 budget and rate figures, and a rate-rollover provision), a MISO Settlements User Group update of 20 July 2026 (administrative content only) and MISO's State of the Market materials of July 2026 (2025 market statistics). No number from any of them was used. The audit record finds that none concerned prices, spreads, uplift amounts, scarcity or weather after June 2026. It lists every document and every query of the re-derivation.
3. **Six CAISO boundary rows.** The July 2026 CAISO downloader stored the 1 July 2026 00:00 PDT hour, which falls in Test A's window, for each of the six CAISO nodes inside the 2026-H1 file.
   - The July 2026 pipeline loaded these rows with the rest of the file, but its analysis windows ended before them.
   - During the revision's forensic review, the file's timestamps were read, which is how the rows were found; their prices were not examined.
   - The revision's development code never reads them: a read-time filter and an assertion guard exclude them, and the sealed downloader's reference reads stop before them.
   - Disclosure, 23 October 2026: one diagnostic script an agent wrote to count hours per node-day read the 2026-H1 development files whole, including these six rows. It printed only per-node-day row counts and counts of non-finite values, never prices, and nothing from those rows was used. The script was then re-run capped at the seal.
4. **Tuning choice.** The rolling-window length W = 56 days was chosen during v1 development, and that is disclosed.
5. **NYISO archive boundary.** NYISO's monthly real-time archive can contain the interval stamped 00:00 on the first day of the next month. The sealed download trims every file to its own settlement days, and logs how many rows were trimmed.
6. **Design decisions taken after the exposure to the ISO-NE proxy figures (item 2).** The exposure happened on 22 October 2026, during the fee research behind the first version of the design spec (spec v1), whose methodology review is stamped 13:46 that day (file times, EDT). Spec v2 (review stamped 14:08), spec v3 and every ruling in the analysis log came after it.

   The exposure could not have informed these decisions. Every hypothesis status is computed by the registered rule from development data, the fees follow a date rule, and no design choice depends on DA–RT differences after June 2026. Examples of decisions taken after it (the list is not complete):
   - the H-E test statistic, its reference set (the full cyclic group of day rotations) and its primary strata;
   - the registration rule;
   - the calibrated fees;
   - the evaluation design of the drift monitors.

   Each of these changes rests on simulation or development-data evidence recorded in the analysis log, except the calibrated fees, which follow the date rule (Rulings 78 and 90). None used any information about prices after June 2026.

## 3. Statement

"I have not examined electricity prices, DART spreads, or market reports for NYISO, ISO-NE, MISO or CAISO covering any date after 30 June 2026, apart from what is stated above. Neither sealed window (Test A: 1 July – 30 September 2026; Test B: 1 October – 31 December 2026) has been downloaded or analysed."

Name and date: Erce Erez, 25 October 2026
