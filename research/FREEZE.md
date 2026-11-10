# Freeze `hpc-freeze-v1`

Branch `research/paper1` is frozen at the commit tagged `hpc-freeze-v1` for HPC deployment. Do not develop on top of
it; new work goes on a new branch from the tag.

**Contents**
- Solver: the received repository plus fix S-1 (`fc9b41d`). No other solver change (S-2 and S-3 are not applied).
- Configurations: `research/campaign/manifests/*.jsonl.gz`, 11 files, 8,394 lines, 7,685 unique run IDs, 0 rejected.
  SHA-256 checksums are in `research/campaign/manifests/FREEZE_checksums.json`; counts are in `SUMMARY.json`.
- Environment: `research/environment/requirements-lock.txt` (Python 3.13.5, numpy 2.1.3, scipy 1.15.3, numba 0.61.0,
  astropy 7.0.0).
- Verification: 41 tests (`python -m pytest research/tests`); evidence in `research/CHANGELOG.md`.

**Integrity check on the cluster**
`python -c "import json,hashlib;[print(e['file'], hashlib.sha256(open('research/campaign/manifests/'+e['file'],'rb').read()).hexdigest()==e['sha256']) for e in json.load(open('research/campaign/manifests/FREEZE_checksums.json'))]"`

**First batch: convergence matrix**
- Manifest: `research/campaign/manifests/V_convergence.jsonl.gz` (gunzip to `.jsonl`).
  - 90 runs: 5 corner states × 9 numerical variants × (tumbler + PA twin).
  - 82 executable: 62 `full` and 20 `full_costly`, about 900 core-h at Apple M4 speed. The 8 `multiscale_required`
    runs are skipped automatically.
- Acceptance criteria: in the docstring of `research/campaign/build_convergence.py`.
