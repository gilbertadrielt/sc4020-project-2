#!/usr/bin/env bash
# Reproduce every number, table and figure.
#
#   bash run_all.sh           full run  (~45 min on 2 cores)
#   bash run_all.sh --quick   DAVIS only, 2 seeds (~5 min)
#
# Each stage reads what the previous one wrote and can be re-run alone.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$PWD/src"

QUICK="${1:-}"
DATASETS="davis kiba"
[ "$QUICK" = "--quick" ] && DATASETS="davis"

echo "=== 0. tests ==========================================================="
python -m pytest -q

echo "=== 1. download raw data ==============================================="
python -m dta.data --download

echo "=== 2. clean, deduplicate, index ======================================="
python -m dta.data $DATASETS

echo "=== 3. freeze splits: 4 regimes x 5 seeds, with leakage asserts ========"
python -m dta.splits $DATASETS

echo "=== 4. kernels and features ============================================"
python -m dta.kernels $DATASETS

echo "=== 5. Task 1: analysis, tables, figures ==============================="
python -m dta.run_task1 $DATASETS

echo "=== 6. tune on validation, evaluate across seeds ======================="
python -m dta.protocol $DATASETS $QUICK

echo "=== 7. buckets, lookup check, case study ==============================="
python -m dta.run_analysis $DATASETS

echo "=== 8. result figures =================================================="
python -m dta.figures $DATASETS
python -m dta.figures_merge $DATASETS

echo "=== 9. LaTeX tables ===================================================="
python -m dta.tables $DATASETS

echo
echo "Done."
echo "  figures -> results/figures/   tables -> results/tables/ (+ tex/)"
echo "  models  -> saved_models/      splits -> data/splits/"
