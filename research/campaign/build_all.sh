#!/usr/bin/env bash
# Rebuild every campaign manifest from its seed (deterministic). Run from the repository root:
#   bash research/campaign/build_all.sh [workers]
# No simulation is run. Seeds: A 11xx (Morris), B 21xx (Sobol'), C = B seeds (first 256 points), D 31xx, E 41xx.
set -euo pipefail
W=${1:-8}
M=research/campaign/manifests
B=(./venv/bin/python research/campaign/build_campaign.py)
mkdir -p "$M"
# resumable: a stage whose manifest (and its CSV) already exists is skipped
run() { local out=${@: -1}; if [ -s "$out" ] && [ -s "${out%.jsonl}.csv" ]; then echo "skip $out"; else "${B[@]}" "$@"; fi; }
run design --stratum slow      --design morris --r 20 --levels 8 --seed 1101 --twins P_par --workers "$W" --out $M/A_morris_slow.jsonl
run design --stratum ordinary  --design morris --r 20 --levels 8 --seed 1102 --twins P_par --workers "$W" --out $M/A_morris_ordinary.jsonl
run design --stratum slow      --design sobol --n 1024 --seed 2101 --twins P_par --workers "$W" --out $M/B_sobol_slow.jsonl
run design --stratum ordinary  --design sobol --n 1024 --seed 2102 --twins P_par --workers "$W" --out $M/B_sobol_ordinary.jsonl
run design --stratum slow      --design sobol --n 256 --seed 2101 --twins P_phi P_lc --workers "$W" --out $M/C_conventions_slow.jsonl
run design --stratum ordinary  --design sobol --n 256 --seed 2102 --twins P_phi P_lc --workers "$W" --out $M/C_conventions_ordinary.jsonl
run phase --source $M/B_sobol_slow.jsonl     --n-states 48 --replicates 8 --seed 3101 --workers "$W" --out $M/D_phase_slow.jsonl
run phase --source $M/B_sobol_ordinary.jsonl --n-states 48 --replicates 8 --seed 3102 --workers "$W" --out $M/D_phase_ordinary.jsonl
run design --stratum fast      --design sobol --n 256 --seed 4101 --twins P_par --workers "$W" --out $M/E_sobol_fast.jsonl
run design --stratum superfast --design sobol --n 256 --seed 4102 --twins P_par --workers "$W" --out $M/E_sobol_superfast.jsonl
echo "ALL MANIFESTS BUILT"
