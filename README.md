# dart-operating-point-drift

Code for the paper **Feature-Block Portability and Operating-Point Drift in DART-Spread Spike Prediction**, by Erce Erez (sole author), written for *Energy Systems*.

## What this is

This is the price-only pipeline of the paper, and nothing else:

- the downloaders of the public day-ahead and real-time prices of four U.S. operators (NYISO, ISO-NE, MISO, CAISO) for a panel of 33 pricing nodes (`v2/ingest_nyiso.py`, `v2/ingest_isone.py`, `v2/ingest_miso.py`, `v2/ingest_caiso.py`) and the settlement legs of CAISO and NYISO (`es/settlement.py`);
- the development pipeline, on data up to 2026-06-30: scores, cutoff policies, economics, drift monitors, inference, and the regeneration of the numbers of the earlier version of the paper;
- the machinery of the two pre-specified sealed tests (Test A, July to September 2026; Test B, October to December 2026): the gate that checks the protocol record (`es.registration`), the sealed pull and the one-shot run;
- the generators of the paper's macros, tables and figures and of the submission package, which read the paper's folder (not part of this package);
- the tests (`tests/es/`).

Apart from the folder `protocol/` (see "How the protocol binds the code"), it holds no price data, raw or derived, no outputs, no manuscript source, no notebooks, and no script of the other projects that share the research repository (text extraction, fundamentals ingestion, ERCOT and PJM probes). `src/` is the shared library of that repository, and some of its comments and configuration entries still mention those projects: the copies are not edited.

## Environment

Python 3.12.7, with these exact versions (read with `importlib.metadata`; `requirements.txt` pins them). They are the versions of the environment that produced the results; other versions are untested.

| package | version |
|---|---|
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| pyarrow | 16.1.0 |
| xgboost | 3.0.2 |
| scikit-learn | 1.5.1 |
| scipy | 1.13.1 |
| statsmodels | 0.14.2 |
| matplotlib | 3.9.2 |

## Layout and setup

The code finds its folders from its own location and from `DART_PROJECT_ROOT` (`es/config.py`, `src/paths.py`). It expects a working folder `W`, called `EnergySystems_Revision` in the research repository (one test checks the name), that holds this package under the name `dart_dev` (`src/paths.py` fixes that name: `v2/cell_builder.py` reads `dart_dev/cache/common_panel_groups.json` through it, and the score cache is written to `dart_dev/cache/`), and an empty folder `DART_Spike_v2_Clean` that both modules still check for (a leftover of the research repository):

```
mkdir EnergySystems_Revision && cd EnergySystems_Revision
git clone <repository url> dart_dev
mkdir DART_Spike_v2_Clean
export DART_PROJECT_ROOT="$PWD"
cd dart_dev
python -m pip install -r requirements.txt
```

Raw prices are read from `~/dart-spread-data/`, outputs go to `dart_dev/outputs/es/`, and the protocol files are read from `W/docs/sealed/`. Run every command below from `dart_dev/` with `DART_PROJECT_ROOT` set.

## Where the data come from

All prices come from the public sites of the four operators, without a key or a login. This package holds none of them; its own code re-pulls them.

| series | source | code |
|---|---|---|
| NYISO day-ahead and real-time zonal prices | mis.nyiso.com/public/csv/ (monthly zips) | `v2/ingest_nyiso.py` |
| ISO-NE day-ahead and final real-time hourly prices | www.iso-ne.com/static-transform/csv/histRpts/ | `v2/ingest_isone.py` |
| MISO day-ahead ex-post and final real-time prices | docs.misoenergy.org/marketreports/ | `v2/ingest_miso.py` |
| CAISO day-ahead and real-time prices | oasis.caiso.com (OASIS) | `v2/ingest_caiso.py` |
| CAISO 15-minute FMM prices and NYISO integrated real-time prices (the settlement legs) | oasis.caiso.com, mis.nyiso.com | `es/settlement.py` |

The panel is the 33 nodes of `v2/NODE_MAP.csv`: 10 NYISO zones, 9 ISO-NE locations, 8 MISO hubs and 6 CAISO nodes; the SPP rows of that file are not part of the panel. CAISO's OASIS keeps a moving window of history, so its oldest requests may no longer be answered and a re-pull may start later than the paper's CAISO series. The development loaders refuse dates after 2026-06-30; the sealed windows open only through the gate that checks the protocol record (`es.registration`).

## Run order

Development period (data to 2026-06-30). Each module's docstring says what it reads and writes.

1. Prices. Run the four downloaders (`python v2/ingest_nyiso.py --start 2023-01 --end 2024-12 --root DIR`; the other three take `--start` and `--end` as dates, and `--help` lists the options) into `~/dart-spread-data/backfill_2023_2024/` for 2023 to 2025 and `~/dart-spread-data/holdout_2026H1/` for 2026-H1 (`DEV_DATA_DIRS` in `es/config.py`). Then the settlement legs, from Python: `es.settlement.pull_caiso_fmm(start, end, root)` with `root` = `~/dart-spread-data/caiso_fmm_dev/` and `es.settlement.pull_nyiso_integrated(start_month, end_month, root)` with `root` = `~/dart-spread-data/nyiso_rtint_dev/`; each needs a root that does not exist yet and refuses an end after 2026-06-30.
2. `python -m es.run_legacy`: regenerates the numbers of the earlier version of the paper and compares them with the printed ones (`es/published_v3.csv`).
3. `python -m es.run_analogs`: quarterly analogues, placebo pairings, the choice of eta, and the size-and-power simulation (`es/simulate.py`). `--stage analogs` and `--stage sim` run its two stages one at a time.
4. `python -m es.run_economics`: the development economic analysis. `python -m es.run_economics_block --block DART3` repeats it with the plain DART lags as the alert model (an exploratory secondary analysis on the development quarters; its files go to `outputs/es/economics_block/DART3/`).
5. `python -m es.run_labels_pr`: precision and recall, and the 2x2 label control.
6. `python -m es.run_monitors`: the label-free drift monitors.
7. `python -m es.run_inference`: mixed model, leave-one-operator-out, multiplicity (needs step 3).
8. `python -m es.run_digest`: one Markdown digest of all outputs.
9. `python -m es.run_driftpath`: the alert rate of the fixed and the rolling cutoff through the two main deployment windows, week by week and over trailing windows (`WINDOW_WEEKS` weeks, the length of the rolling cutoff's window), written to `outputs/es/paper/drift_path.csv` and `outputs/es/paper/drift_summary.csv`. It needs the cached scores (it reads the score cache and never refits a model: with a cold cache it stops) and checks its counts against the outputs of steps 2 and 4. The macros and `es.paper_figures` read its two files, so it runs before step 10.
10. The paper generators, which need the paper's folder (`paper/` beside `dart_dev/`): `python -m es.impl_table`, `python -m es.paper_macros`, `python -m es.paper_tables`, `python -m es.paper_figures`, `python -m es.paper_lint`, and `python -m es.paper_package` for the journal upload set.

Sealed tests (see the next section for the ones that have run):

11. `python -m es.protocol` writes the protocol package into `W/docs/sealed/` (pre-pull cutoffs, protocol inputs, `CODE_MANIFEST.sha256`, `code_archive.tar.gz`, `ENV_PINS.json`, `PROTOCOL.md` and `PROTOCOL.sha256`). It renders the protocol from the design spec in `W/docs/superpowers/specs/`, which is not part of this package. The protocol and its files are copied to `protocol/` in this repository, and so are the records of the sealed tests; the code reads `W/docs/sealed/`, not `protocol/`. Steps 12 and 13 run only when a record, `W/docs/sealed/REGISTRATION.md`, names the SHA-256 of `PROTOCOL.md`. The record in `protocol/REGISTRATION.md` does.
12. `es.sealed_pull.pull_sealed("A", "<protocol sha256>")` downloads the July to September 2026 window, once, and only when `es.registration.verify_registration` passes.
13. `es.sealed_run.run_sealed(es.sealed_run.default_config("A"), "<protocol sha256>")` runs Test A once. Test B (October to December 2026) is the same call with `"B"`; it pools Test A's stored rows and needs Test A's completed run.

## How the protocol binds the code

The protocol is in `protocol/`, with its files: `PROTOCOL.md`, `PROTOCOL.sha256`, `CODE_MANIFEST.sha256`, `ENV_PINS.json`, `KNOWLEDGE_STATEMENT.md`, `protocol_inputs.json` and `pre_pull_cutoffs.csv`; the archive `code_archive.tar.gz` is not published (see Redactions). The record of the protocol is `protocol/REGISTRATION.md`. It is a local record, `author-statement-2026-10-26`, recorded at 2026-10-26T08:33:52Z (UTC). The record is a local statement that an AI assistant wrote on the author's instruction; the author did not write or sign it. No registry, repository or other third party held the protocol before the data of the window were pulled, so for the order of events a reader has only that record and the logs of the pull and the run. The protocol is published here with the code after the run. The records of the pulls and the runs are there too: `PULLS.json`, `RUNS.json` and `RUN_A.claim`. The pull of each window is recorded in `PULLS.json` with its start and finish times.

The sealed windows are Test A (1 July to 30 September 2026) and Test B (1 October to 31 December 2026). This is what the records in `protocol/` show about them (no result is quoted here):

- Test A (1 July to 30 September 2026): started at 2026-10-26T09:23:04.872954+00:00 (UTC). It finished at 2026-10-26T09:38:39.473712+00:00 (UTC). Its record is in `protocol/RUNS.json`, and its claim file is `protocol/RUN_A.claim`.
- Test B (1 October to 31 December 2026): not started. There is no record of it in `protocol/RUNS.json`, and there is no `protocol/RUN_B.claim`.

The files that the runs wrote are in `protocol/sealed_outputs/A/`. The `.parquet` files of the runs are not part of this repository.

The logs of the pulls are in `protocol/sealed_pulls/`: `PULL_LOG.txt` gives the start and finish times, the record the pull ran under and the days that were excused; `COVERAGE.csv` and `OVERLAP_CHECK.csv` give the coverage of each node and the check of the first days against the files already opened. `MANIFEST.sha256` in each folder lists the files of that pull, which are not part of this repository.

The sealed tests are bound to the code by content:

- `python -m es.protocol` lists the code that the protocol fixes in `CODE_MANIFEST.sha256`, one `sha256  path` line per file, paths relative to `dart_dev/`: the import closure of `es.sealed_run`, `es.sealed_pull`, `es.registration` and `es.protocol`, traced statically, and the two data files that closure opens by a fixed path (`v2/NODE_MAP.csv` and `cache/common_panel_groups.json`). `PROTOCOL.md` prints the SHA-256 of the manifest, of `code_archive.tar.gz`, of `ENV_PINS.json`, of the pre-pull cutoffs and of the protocol inputs.
- `es.registration.verify_registration` opens a sealed window only when `W/docs/sealed/REGISTRATION.md` exists and names the SHA-256 of the unchanged `PROTOCOL.md`, and `es.registration.verify_bound` checks that every bound file matches the hash the protocol prints.
- `es.sealed_run.code_departures(manifest, root)` lists the files that differ from the manifest of the protocol. The sealed run is refused when there is one, unless the departure is recorded (`allow_departure=True`).

To check the protocol, the record and the code, from the root of this repository (`dart_dev/` once it is cloned as above):

- `shasum -a 256 protocol/PROTOCOL.md` prints `3efce3a1e6fd23c709b738409e5be08da6735811cf0fb104fe17f2ae74f0a7eb`. This is the content of `protocol/PROTOCOL.sha256` and the `protocol_sha256` of `protocol/REGISTRATION.md`.
- `shasum -a 256 -c protocol/CODE_MANIFEST.sha256` (or `sha256sum -c`) checks the code that the protocol fixes, file by file. In Python, `es.sealed_run.code_departures("protocol/CODE_MANIFEST.sha256", ".")` lists the files that differ, and returns `[]` when none does.

The files of this package are copied byte for byte, so the SHA-256 of a file here is that of the file the protocol lists. When this package was built, every line of `protocol/CODE_MANIFEST.sha256` matched the file of the same path in this package, except the line of `src/paths.py` (see Redactions).

## Tests

```
pytest tests/es -m "not slow" -q
```

The `slow` marker is for the tests that run the full 33-node pipeline. No test reaches a network (`tests/es/conftest.py` closes the sockets).

**Fast tests in a clean copy.** On 2026-10-26, `pytest tests/es -m "not slow" -k "not rehearsal"` ran 3,071 tests in about six and a half minutes (no network, empty home folder) in a copy of this package with no raw prices, no `paper/`, no `docs/` and no development outputs: 2,807 passed, 91 failed, 9 errored at setup and 164 were skipped (39 more tests were deselected as slow or as rehearsals). Of the 164 skipped tests, 136 read the manuscript (each says "the manuscript folder is not next to dart_dev") and 28 need an output or a built file that is not in this package (each says which). Each of the other 100 needs a file that is not in this package:

| what is missing | tests | outcome |
|---|---|---|
| the raw price files in `~/dart-spread-data/` (step 1 of the run order) | `tests/es/test_data.py` (5), `tests/es/test_data_seal.py` (1), `tests/es/test_scores.py` (3) | 9 errors |
| the design spec in `W/docs/superpowers/specs/` (not published with this package; the protocol itself is in `protocol/`) | `tests/es/test_protocol.py` (68 of its 77) | fail |
| `W/docs/sealed/KNOWLEDGE_STATEMENT_TEMPLATE.md` | `tests/es/test_protocol.py` (7) | fail |
| `dart_dev/outputs/es/analogs/protocol_inputs.json`, written by `python -m es.run_analogs` | `tests/es/test_protocol.py` (2) | fail |
| the paper's folder `W/paper/` (`build.sh`) | `tests/es/test_paper_build.py` (10) | fail |
| `dart_dev/outputs/es/paper/implementation.json` and the sections of `W/paper/` | `tests/es/test_impl_table.py` (1) | fail |
| `v2/scripts/open_holdout.py`, the July script that holds the original `nwd` function; it is not in this package | `tests/es/test_run_monitors.py` (2, `test_nwd_*`) | fail |
| a working folder called `EnergySystems_Revision` | `tests/es/test_config.py` (1, `test_project_root_resolves_locally`) | fail |

Every other test passes without data. The run had no data in place, so the effect of the raw prices on the errors was not tested, and the skipped manuscript tests were not run against a manuscript in this copy (they were run in the research tree, where the whole fast suite of `tests/es`, 3,109 tests, passes with one skipped).

## Verifying this package

`MANIFEST.sha256` lists every file of the package except itself, one `sha256  path` line each: `shasum -a 256 -c MANIFEST.sha256` (or `sha256sum -c MANIFEST.sha256`), or `python -m es.public_package --verify .` from `dart_dev/`, which also reports files that the manifest does not list (for example `__pycache__` left by running the tests inside the package folder).

## Redactions

Two things differ from the files that the protocol fixes, and nothing else does.

1. `src/paths.py` is published with the author's local Google Drive path (a user name, an account address and the identifier of a shared folder) replaced by the placeholders `<user>`, `<account>` and `<folder-id>`. The file lists the places where the code looks for the research folder, it holds no logic of the analyses, and a reader has to enter the path of their own machine there anyway. `protocol/CODE_MANIFEST.sha256` lists the SHA-256 of the original file, so that one line of the manifest does not match this file: `shasum -a 256 -c protocol/CODE_MANIFEST.sha256` reports it and `es.sealed_run.code_departures` lists `src/paths.py`. Every other line of the manifest matches.
2. `protocol/code_archive.tar.gz`, the deterministic archive of the code that the protocol fixes, is not published, because it contains the original `src/paths.py`. `PROTOCOL.md` prints its SHA-256. The author's own copy of the code and of the archive is unchanged.

The records of the pulls and of the run (`protocol/PULLS.json`, `protocol/RUNS.json`, `protocol/RUN_A.claim`, `protocol/sealed_pulls/` and `protocol/sealed_outputs/`) and every other file of `protocol/` are published byte for byte.

## Licence and citation

MIT (see `LICENSE`), copyright 2026 Erce Erez.

Cite the paper: Erce Erez, *Feature-Block Portability and Operating-Point Drift in DART-Spread Spike Prediction* (DOI to be added).
