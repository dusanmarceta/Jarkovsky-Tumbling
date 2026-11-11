"""
Collect the validation results into one JSON summary (research/runs/superfast_validation/VALIDATION.json):
IAC reference comparison, overlap comparison with the full solver (drift, tumbling effect, cost), convergence
studies and the super-fast demonstration.

    python -m superfast.validation.report
"""
import json
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[2]
for p in (str(RESEARCH), str(RESEARCH.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.manifest import accepted, read_manifest  # noqa: E402

V = RESEARCH / 'runs' / 'superfast_validation'


def overlap(manifest, root, new_file):
    lines = read_manifest(manifest)
    new = json.load(open(new_file)) if Path(new_file).exists() else {}
    rows = {}
    for ln in lines:
        p = Path(root) / 'runs' / ln['run_id'] / 'provenance.json'
        full = json.load(open(p)) if p.exists() else None
        n = new.get(ln['run_id'])
        r = (full or {}).get('result') or {}
        row = {'run_id': ln['run_id'], 'kind': ln['kind'], 'note': ln.get('validation_note'), 'stratum': ln['stratum'],
               'mode': ln['derived']['mode'], 'P_phi_h': ln['inputs']['P_phi_h'], 'ecc': ln['inputs']['ecc'],
               'Gamma': ln['inputs']['thermal_inertia'], 'D_m': ln['inputs']['D_eq_m'],
               'full_cost_estimate_core_h': ln['policy']['cost_core_h_estimate'],
               'full_dadt': r.get('drift_m_s'), 'full_accepted': accepted(full) if full else None,
               'full_wall_s': (full or {}).get('wall_time_s'), 'full_orbits': r.get('orbits'),
               'full_history': r.get('drift_history_m_s')}
        if n:
            row.update(new_dadt=n['dadt_m_s'], new_iterations=n['iterations'], new_cpu_s=n.get('cpu_s'),
                       new_wall_s=n['wall_s'], new_converged=n['all_converged'], new_fast_steps=n['fast_steps'])
            if row['full_dadt']:
                row['rel_diff'] = n['dadt_m_s'] / row['full_dadt'] - 1
            if row.get('full_wall_s') and n.get('cpu_s'):
                row['speedup_core_time'] = row['full_wall_s'] / n['cpu_s']
        if row['full_history'] and len(row['full_history']) > 1:
            h = row['full_history']
            row['full_last_orbit_change_rel'] = abs(h[-1] - h[-2]) / abs(h[-1])
        rows.setdefault(ln['state_id'], {})[ln['kind']] = row
    states = []
    for sid, d in rows.items():
        t, p = d.get('tumbler', {}), d.get('pa_P_par', {})
        s = {'state_id': sid, 'tumbler': t, 'twin': p}
        if all(x.get(k) is not None for x in (t, p) for k in ('full_dadt', 'new_dadt')):
            s['delta_full'] = t['full_dadt'] - p['full_dadt']
            s['delta_new'] = t['new_dadt'] - p['new_dadt']
            s['delta_diff_over_twin'] = (s['delta_new'] - s['delta_full']) / abs(p['full_dadt'])
        states.append(s)
    return states


if __name__ == '__main__':
    out = {'iac_compare': json.load(open(V / 'iac_compare.json')) if (V / 'iac_compare.json').exists() else None,
           'overlap': overlap(V / 'overlap.jsonl', V / 'raw' / 'full', V / 'new_default.json'),
           'convergence': json.load(open(V / 'convergence.json')) if (V / 'convergence.json').exists() else None,
           'superfast_demo': json.load(open(V / 'superfast_demo_new.json')) if (V / 'superfast_demo_new.json').exists() else None}
    json.dump(out, open(V / 'VALIDATION.json', 'w'), indent=1, default=float)
    for s in out['overlap']:
        t, p = s['tumbler'], s['twin']
        f = lambda x: 'n/a' if x is None else f'{x:+.2e}'
        print(f"{s['state_id']:28s} tumbler rel {f(t.get('rel_diff'))}  twin rel {f(p.get('rel_diff'))}  "
              f"Delta diff/twin {f(s.get('delta_diff_over_twin'))}  speed-up {t.get('speedup_core_time') or 0:.0f}x")
