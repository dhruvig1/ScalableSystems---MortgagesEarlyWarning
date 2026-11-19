#!/usr/bin/env bash
# Fail if anything that looks like a key, password or connection string is in the repo.
# Run before every push and before zipping:  bash scripts/secret_scan.sh
cd "$(dirname "$0")/.."
PATTERNS='mongodb(\+srv)?://[^<$ "'"'"']+:[^<$ "'"'"']+@|sk-ant-[A-Za-z0-9_-]{10,}|AIza[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|(api[_-]?key|password|secret|token)["'"'"' ]*[:=]["'"'"' ]*[A-Za-z0-9_\-]{20,}'
HITS=$(grep -rEn --exclude-dir=.git --exclude-dir=data --exclude-dir=__pycache__ --exclude-dir=.pytest_cache \
        --exclude=secret_scan.sh -i "$PATTERNS" . 2>/dev/null | grep -v "os.environ" || true)
if [ -n "$HITS" ]; then echo "POSSIBLE SECRETS:"; echo "$HITS"; exit 1; fi
if git ls-files --error-unmatch .env >/dev/null 2>&1; then echo ".env is tracked!"; exit 1; fi
echo "secret scan clean"
