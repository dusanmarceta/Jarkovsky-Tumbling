"""
Collect finished runs of one or more manifests into a single analysis table (CSV), one row per manifest line:
all inputs, derived rotational quantities, thermal/validity quantities, policy and numerics, run status,
drift, per-orbit drift history (JSON), diagnostics, wall time, code commit and environment.
Also prints a failure report (missing, failed, invalid, not converged, fallback, temperature bound).

    python research/hpc/aggregate.py --manifest M1.jsonl [M2.jsonl ...] --root R1 [R2 ...] --out results.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
from campaign.build_campaign import flatten  # noqa: E402
from tools.manifest import accepted, read_manifest  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', nargs='+', required=True)
    ap.add_argument('--root', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows, report = [], {}
    for man, root in zip(a.manifest, a.root):
        for i, ln in enumerate(read_manifest(man)):
            row = flatten({k: v for k, v in ln.items() if k != 'config_overrides'})
            row.update({'manifest': Path(man).name, 'line': i})
            p = Path(root) / 'runs' / ln.get('run_id', '-') / 'provenance.json'
            if ln['kind'] == 'rejected':
                status = 'rejected'
            elif not p.exists():
                status = 'missing'
            else:
                rec = json.load(open(p))
                r = rec.get('result') or {}
                row.update({f'result.{k}': (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
                row.update({'wall_time_s': rec.get('wall_time_s'), 'exit_code': rec.get('exit_code'),
                            'git_commit': rec.get('git', {}).get('commit'), 'numpy': rec.get('environment', {}).get('numpy'),
                            'numba': rec.get('environment', {}).get('numba'), 'attempt': (rec.get('inputs') or {}).get('attempt')})
                if accepted(rec):
                    status = 'accepted'
                elif rec.get('exit_code') != 0 or not r:
                    status = 'failed'
                elif not r.get('valid'):
                    status = 'invalid'
                elif r.get('fallback_steps', 0) > 0:
                    status = 'fallback'
                elif not r.get('converged'):
                    status = 'not_converged'
                else:
                    status = 'temperature_bound'
            row['status'] = status
            report[status] = report.get(status, 0) + 1
            rows.append(row)
    keys = sorted({k for r in rows for k in r})
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({'rows': len(rows), 'status': report, 'out': a.out}, indent=1))
