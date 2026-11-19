# How Claude was used and controlled

Claude appears in this project in three places. In each, the rules below are what limits it.

## 1. Runtime: labeling complaints (`src/ai/label_complaints.py`, `src/ai/themes.py`)
- Model: `claude-haiku-5-5`, called once per complaint narrative (first 1,500 characters), 3,000 complaints.
- Instruction (the `SYSTEM` string): classify into exactly one theme from a fixed list, reply with only
  `{"theme": ..., "confidence": ...}`. The six themes and their definitions live in `src/ai/themes.py`:
  payment_hardship, servicing_error, escrow, modification_denial, foreclosure, origination_other.
- Guardrails in code: the reply is parsed as JSON and rejected unless the theme is in the allowed list
  (`parse_reply`); failures retry, then the row is skipped, never guessed. Labels are written to a CSV
  first and pushed to Atlas only with `--write-mongo`. Existing labels are not overwritten on reload.
- Checking it: a person who did not know a model was involved labeled 70 complaints blind
  (`src/ai/hand_label_cli.py`, results in `results/hand_labels.csv`). Agreement 65.7%, kappa 0.556
  (`src/ai/validate_labels.py`). A second check against CFPB's own issue categories gave 46%.
  Themes overlap, so they are a search aid, not a measurement.

## 2. Runtime: Claude Desktop over MongoDB MCP (`mcp/claude_desktop_config.template.json`)
- Server: `mongodb-mcp-server` started with `--readOnly`, using a database user that can only read.
- The connection string is set only in the local Claude Desktop config, never in the repo.
- Used for demo questions such as counts by theme and the scenario with the highest expected loss.
  The same numbers are verified elsewhere (`results/RESULTS.md`).

## 3. Build time: Claude as coding assistant
- Claude wrote and edited the code in this repo at my direction; I ran it, read the outputs and decided
  what to change. Every result number in the README, deck and `results/RESULTS.md` comes from a run I
  executed on my machine, the VM or Dataproc, not from Claude's memory.
- Standing rules I gave it: never print or commit keys, passwords or connection strings
  (`scripts/secret_scan.sh` is run before every push); report limits and failures instead of hiding them
  (for example the hardship 3-month model that did not converge, and the assumed 35% loss given default);
  no invented published numbers (`docs/data_reconciliation.md` records that Freddie publishes no
  per-quarter counts).
- Mistakes Claude made that testing caught: wrong vintage year for loans originated in the 1990s, an
  SDK decoder error, and reading model "thinking" blocks as text. Each was fixed and re-run.
