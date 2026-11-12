#!/usr/bin/env bash
# Reproduce every questionnaire answer from the raw data.
#
#   conda activate con2phys
#   bash reproduce.sh               # one animal at a time (several hours)
#   N_JOBS=8 bash reproduce.sh      # 8 animals in parallel (needs ~8 GB RAM each)
#
# For each question in config.yaml, every animal is analysed
# (python -m questions.<q> --animal N), then the animals are combined into
# outputs/<q>/final_report.json (answer, mean and 95% CI per condition, tests)
# with figures in outputs/<q>/figures/.
set -euo pipefail
cd "$(dirname "$0")"

N_JOBS="${N_JOBS:-1}"
QUESTIONS=$(python -c 'import yaml; print(" ".join(yaml.safe_load(open("config.yaml"))["questions"]))')
ANIMALS=$(python -c 'from con2phys.io import included_mice, read_config
print(" ".join(map(str, included_mice(read_config("config.yaml")))))')
echo "animals:   $ANIMALS"
echo "questions: $QUESTIONS"

for question in $QUESTIONS; do
    echo "=== $question: per-animal analysis"
    printf '%s\n' $ANIMALS | xargs -P "$N_JOBS" -I{} python -m "questions.$question" --animal {}
    echo "=== $question: across-animal statistics"
    python -m "questions.$question" --aggregate
done
echo "Done. Answers: outputs/<question>/final_report.json and final_report.md"
