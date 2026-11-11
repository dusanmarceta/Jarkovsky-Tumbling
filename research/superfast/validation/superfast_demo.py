"""
Step 3: the two-scale method on currently unsupported campaign runs (super-fast stratum and non-executable fast
runs): cost per run against the full solver's cost estimate, convergence, and physical diagnostics.

    python -m superfast.validation.superfast_demo select --out runs/superfast_validation/superfast_demo.jsonl
    python -m superfast.validation.overlap new --manifest runs/superfast_validation/superfast_demo.jsonl --root R --out ...
"""
import argparse
import json
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[2]
for p in (str(RESEARCH), str(RESEARCH.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from campaign import waves as W  # noqa: E402


def select(out):
    frozen = W.load_frozen()
    waves = W.assign(frozen, {})
    deferred = [ln for ln, _, _ in waves['deferred'].values()]
    tw = {ln['state_id']: ln for ln in deferred if ln['kind'] == 'pa_P_par'}
    tum = [ln for ln in deferred if ln['kind'] == 'tumbler' and ln['state_id'] in tw]
    sf = sorted([l for l in tum if l['stratum'] == 'superfast'], key=lambda l: l['inputs']['P_phi_h'])
    fa = sorted([l for l in tum if l['stratum'] == 'fast'], key=lambda l: l['policy']['cost_core_h_estimate'])
    picks = {
        'superfast shortest P_phi': sf[0],
        'superfast 10th percentile P_phi': sf[len(sf) // 10],
        'superfast median P_phi': sf[len(sf) // 2],
        'superfast LAM, highest e': max((l for l in sf if l['derived']['mode'] == 'LAM'), key=lambda l: l['inputs']['ecc']),
        'superfast lowest Gamma': min(sf, key=lambda l: l['inputs']['thermal_inertia']),
        'superfast smallest D': min(sf, key=lambda l: l['inputs']['D_eq_m']),
        'fast non-executable, cheapest': fa[0],
        'fast non-executable, most expensive': fa[-1],
    }
    lines = []
    for note, t in picks.items():
        for ln in (t, tw[t['state_id']]):
            lines.append(dict(ln, validation_note=note))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        for ln in lines:
            f.write(json.dumps(ln, default=float) + '\n')
    for ln in lines[::2]:
        i, d = ln['inputs'], ln['derived']
        print(f"{ln['validation_note']:38s} {ln['state_id']} {d['mode']} P_phi {i['P_phi_h'] * 3600:8.1f} s "
              f"e {i['ecc']:.2f} Gamma {i['thermal_inertia']:6.0f} D {i['D_eq_m']:6.1f} m  full-solver estimate "
              f"{ln['policy']['cost_core_h_estimate']:.3g} core-h")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select')
    s.add_argument('--out', required=True)
    a = ap.parse_args()
    select(a.out)
