# Demo script (about 10 minutes total, demos ~4 minutes)

## Setup before recording
1. Copy `mcp/claude_desktop_config.template.json` into Claude Desktop's config on your machine
   and paste the Atlas connection string there (that file lives outside the repo). Keep
   `--readOnly`. Restart Claude Desktop. Confirm the MongoDB tools show up. Check the flag names
   against the current mongodb-mcp-server README.
2. Make sure `complaints` has `theme` filled for the labeled sample and the Atlas Search index
   `complaints_search` is READY.
3. Hide the terminal env and the Atlas UI connection string. No secrets on screen.

## Demo 1: Claude over the MongoDB MCP server (about 2 min)
Prompts to type live:
- "What collections are in the mew database and how many documents does each have?"
- "Using complaints with a theme, which five states had the largest increase in payment_hardship
  complaints in the last 12 months versus the prior 12?"
- "Cross-check: use state_month_rollup to give those states' 90+ day delinquency rate for the
  same two periods. Does delinquency rise after the complaints?"
Say out loud: Claude queries MongoDB through the MCP server; I verify the same numbers in the
dashboard and in the lead-lag results.

## Demo 2: Atlas Search (about 1 min)
`python -m src.serve.search_demo --query "lost job forbearance ended" --state TX`
Show ranked hits and the facet counts by issue.

## Demo 3: Dashboard (about 1.5 min)
`streamlit run app/dashboard.py -- --outputs data/outputs --models data/models`
Walk the tabs: risk model + calibration, state delinquency + the "same number three ways" table,
lead-lag, scenarios.

## Demo 4: Pipeline proof (about 30 s)
Show the Spark job finished on GKE/Dataproc (console screenshot or terminal) and the Airflow DAG graph.

## Close on
- Answer to each of the 3 questions, with the number and its caveat.
- How much data (rows, GB) and why one machine cannot hold it.
- What the AI step got right and wrong (accuracy vs hand labels).

## Added after instructor feedback
- Demo 5 (about 1 min): show `src/features/spark_panel.py` window definitions and `docs/queries.md`.
- Demo 6 (about 30 s): `python -m src.ingest.cfpb_live --dry-run` to show new complaint counts from CFPB's daily API (metadata only).
- Show `docs/AGENTS.md` (how Claude was controlled).
- End with a 30 s repo walkthrough: README, `src/` by stage, `docs/`, `results/`, `dags/`, `infra/`.

- Hosted dashboard (backup if the local one fails): https://scalablesystems---mortgagesearlywarning-7selzvawr4hojizmvkg28m.streamlit.app/
