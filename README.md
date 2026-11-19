# Mortgage Credit Early-Warning

Predict and stress-test 12-month serious delinquency (90+ days late) on the full Freddie Mac
loan-level history, joined to FRED macro data and CFPB mortgage complaints.

## 1. The problem

**Who:** mortgage servicers, banks and investors in mortgage bonds. **Decision:** how much to
reserve for losses (CECL), which borrowers to contact first, and what a rise in unemployment or
rates would do to the book. Today they learn late, because reports flag a loan only after it is
30/60/90 days behind. Analysts also tend to use a sample or one year, which hides the 2008 and
2020 regimes and regional differences.

**Questions**
1. Which loans will be 90+ days late within 12 months, using only what was known at the time?
2. Do jumps in a state's CFPB mortgage complaints show up *before* its delinquency rises?
3. How much would portfolio delinquency and expected loss move if unemployment or rates rose?

**A good answer** is a calibrated risk ranking, a tested early-warning signal, and loss ranges
per scenario. A risk team can reserve the right amount and target outreach where it prevents defaults.

## 2. Architecture

![architecture](docs/architecture.svg)

| Stage | Tool | Job | Why this tool |
|---|---|---|---|
| Ingest + store | DuckDB -> Parquet on GCS | Convert headerless pipe files to typed, partitioned columnar files; EDA and window-function SQL | Columnar, compressed, fast single-machine SQL; Parquet is the shared format for every engine |
| Transform | Spark on GKE / Dataproc | Build the loan-month panel (12-month look-ahead label, lags, rolling windows, macro join) | Billions of rows; every row of a loan must meet on one executor, so we shuffle by `loan_id` and scale out with executors |
| Model | LightGBM + logistic baseline | Risk score with calibration and subgroup checks | Gradient boosting handles mixed and missing features; logistic is the honest baseline |
| Documents + search | MongoDB Atlas + Atlas Search | Complaints (free text, new fields as Claude labels them), rollups, scored loans, scenarios | Variable-shape documents and full-text search over narratives |
| AI | Claude (API batch labeling + MongoDB MCP server) | Theme labels on a capped sample; ad hoc analyst questions | Checked against CFPB issue categories and a hand-labeled sample |
| Orchestrate | Airflow (`dags/mew_refresh.py`) | Monthly refresh | Dependencies and reruns |
| Serve | Streamlit (`app/dashboard.py`) | Results for the demo | |

### Data layers (bronze / silver / gold)

| Layer | Contents | Location | Built by |
|---|---|---|---|
| Bronze (raw, never edited) | Freddie yearly zips, CFPB archive zips, raw FRED pulls | `gs://<bucket>/raw/` | manual download, `fred_download.py` |
| Silver (cleaned, typed) | Freddie origination + performance Parquet (types cast, coded unknowns such as 9999 set to NULL, `dq_months` parsed, partitioned by vintage); de-duplicated CFPB mortgage complaints; lagged state-month macro table | `gs://<bucket>/silver/` | `freddie_to_parquet.py`, `cfpb_to_parquet.py`, `build_macro.py` (DuckDB) |
| Gold (built for a question) | Loan-month panel with the 12-month label and features; state-month delinquency rollups; scored loans; scenario results; complaint themes; Atlas collections; dashboard inputs | `gs://<bucket>/gold/`, MongoDB Atlas | `spark_panel.py` (Spark), `train.py`, `score.py`, `monte_carlo.py`, `label_complaints.py` |

Bronze is immutable so any later layer can be rebuilt from it. Quality work (layout changes, coded
values) happens at bronze to silver; modeling and serving happen at silver to gold.

## 3. Data (not in the repo)

| Source | How to get it | Size |
|---|---|---|
| Freddie Mac Single-Family Loan-Level Dataset, 1999-2026 | Free registration at Freddie Mac Clarity; download the yearly zips by hand into `data/raw/freddie/` | ~56M loans, ~40 GB zipped, billions of loan-month rows |
| CFPB Consumer Complaint Database | CFPB complaint archive page; 21 zips into `data/raw/cfpb/` | 459,836 mortgage complaints, 147,249 with narrative |
| FRED | Free API key; `export FRED_API_KEY=...` | Small (state unemployment, house price index, 30-year rate) |

Hard parts and how they are handled: headerless files and per-year layout changes (layout detected
per file and logged), coded unknowns like 9999 (set to NULL, never imputed), rare events and open
loans (fixed 12-month label, last 12 months unlabeled, time split with a 12-month embargo),
location join only at state-month level, CFPB narratives stop 14 Aug 2026.

**Big-data rule:** the full Freddie history is far larger than a laptop's memory and disk. Spark on
a cluster builds the panel; DuckDB/Parquet does the exploration.

## 4. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # Java 11/17 needed for local pyspark
cp .env.example .env                       # fill in locally; .env is gitignored
set -a; source .env; set +a                # exports FRED_API_KEY, MONGODB_URI, ANTHROPIC_API_KEY
pytest -q                                  # synthetic-data tests, no real data or keys needed
```

## 5. Run the pipeline

```bash
# 1. Ingest (run on a VM with a big disk; resumable, skips finished quarters)
python -m src.ingest.freddie_to_parquet --zips data/raw/freddie/*.zip --out data/parquet --tmp-dir /mnt/tmp
python -m src.ingest.cfpb_to_parquet --inputs data/raw/cfpb/*.zip --out data/parquet/cfpb
python -m src.ingest.fred_download --out data/parquet/fred
python -m src.features.build_macro --fred data/parquet/fred/fred_series.parquet \
       --out data/parquet/macro/state_month.parquet
gsutil -m rsync -r data/parquet $BUCKET/silver

# 2. EDA (DuckDB over Parquet)
python -m src.eda.profile --parquet data/parquet --cfpb data/parquet/cfpb/mortgage_complaints.parquet \
       --out data/outputs/eda

# 3. Panel on the cluster (Spark)
bash infra/submit_dataproc.sh panel          # or: bash infra/submit_gke.sh panel
gsutil -m rsync -r $BUCKET/gold/panel data/panel_spark
#    single-machine reference (same output, use on a slice): python -m src.features.build_panel ...

# 4. Model, scoring, scenarios
python -m src.models.train --panel data/panel_spark --out data/models --split time --sample-frac 0.1
python -m src.models.train --panel data/panel_spark --out data/models_nomacro --no-macro   # ablation
python -m src.models.score --panel data/panel_spark --models data/models \
       --out data/outputs/portfolio_snapshot.parquet
python -m src.scenarios.monte_carlo --snapshot data/outputs/portfolio_snapshot.parquet \
       --models data/models --out data/outputs

# 5. Atlas + Claude
python -m src.serve.mongo_load complaints --parquet data/parquet/cfpb/mortgage_complaints.parquet
python -m src.serve.mongo_load indexes                       # Atlas Search index
python -m src.ai.label_complaints --parquet data/parquet/cfpb/mortgage_complaints.parquet \
       --out data/labels --n 3000 --write-mongo
python -m src.ai.validate_labels cfpb --parquet data/parquet/cfpb/mortgage_complaints.parquet \
       --labels data/labels/claude_labels.csv --out data/outputs
python -m src.ai.validate_labels export-hand --parquet data/parquet/cfpb/mortgage_complaints.parquet \
       --labels data/labels/claude_labels.csv --out data/labels      # label 200 by hand, then:
python -m src.ai.validate_labels hand --labels data/labels/claude_labels.csv \
       --hand data/labels/hand_labels.csv --out data/outputs

# 6. Ask it twice: the same number in DuckDB, Spark SQL and MongoDB (on a VM slice or full copy)
python -m src.queries.state_quarter --parquet data/parquet --out data/outputs --spark --mongo
python -m src.queries.loan_counts --parquet data/parquet --published docs/freddie_published_counts.csv

# 7. Question 2 and the dashboard
python -m src.analysis.complaints_lead_lag --rollup data/outputs/state_month_rollup.parquet \
       --complaints data/parquet/cfpb/mortgage_complaints.parquet --out data/outputs
streamlit run app/dashboard.py -- --outputs data/outputs --models data/models
```

For the interactive MCP demo see `mcp/claude_desktop_config.template.json` and `docs/demo_script.md`.

## 6. How results are checked

- **Same number, several ways:** state-by-quarter 90+ day rate in DuckDB SQL (raw Parquet), Spark SQL
  (raw Parquet), and a MongoDB aggregation over the stored state-month rollup. `state_quarter.py`
  exits non-zero unless counts match exactly. Loan counts are compared with Freddie's published totals.
- **Spark vs DuckDB panel:** `tests/test_spark_equivalence.py` asserts identical rows, labels and features.
- **No leakage:** features use rows at or before month t; macro series are lagged by their release
  delay (`build_macro.py`); time split enforces a >12-month embargo; train/validation split is by loan.
- **Calibration and groups:** `train.py` reports AUC, average precision, Brier, ECE, isotonic
  recalibration, and metrics by credit band, LTV, DTI, occupancy, channel, vintage and year.
- **AI output:** Claude's themes are scored against CFPB's own issue categories (rule mapping) and a
  200-complaint hand-labeled sample (accuracy and Cohen's kappa).
- **Q2 lead-lag:** state and month fixed effects, clustered errors, controls for delinquency
  momentum, and a placebo that shuffles complaints across states.
- **Q3 scenarios:** the model re-scores the current book under shocked macro inputs. This shows what
  the model predicts, not a causal effect. LGD is an assumption set by `--lgd`.

## 7. Results

_Fill in after the full runs (numbers, not adjectives):_ rows and GB processed; Spark job runtime
and executors; AUC / ECE for logistic vs LightGBM, with and without macro; the three-way check
output; Q2 coefficients by horizon vs placebo; Q3 scenario table; Claude label accuracy.

## 8. Limits

Complaints cover all lenders while loans are Freddie Mac only, so they join at state-month level.
Narratives are about a third of complaints and end Aug 2026. Scenario results inherit the model's
learned macro sensitivity. The 2020 forbearance period changes what delinquency means; report by period.

## 9. Security

No keys or connection strings in this repo. `FRED_API_KEY`, `MONGODB_URI`, `ANTHROPIC_API_KEY` are read
from the environment only. Run `bash scripts/secret_scan.sh` before pushing, zipping, or recording.
