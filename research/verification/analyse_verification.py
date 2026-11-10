"""
Evaluate the solver-level verification runs against their acceptance criteria.

  scaling      ratios must equal 2, 2 and 4 to 1e-9 (identical numerics, 2 orbits each)
  convergence  each discretisation change relative to the base setting; Richardson estimate of the
               remaining error for steps and dz; production setting accepted if every error < 0.5 %
               (or below the phase-noise floor, whichever is larger)
  phase        spread of the converged drift over initial phases phi_0 (and lambda_L at e = 0)
Writes research/runs/verification/summary.json and prints a table.
"""
import json
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[1] / 'runs' / 'verification' / 'raw'


def load(name):
    p = RAW / name / 'provenance.json'
    if not p.exists():
        return None
    rec = json.load(open(p))
    return rec if rec.get('result') else None


def drift(name):
    r = load(name)
    return None if r is None else r['result']['drift_m_s']


def rel(a, b):
    return None if a is None or b is None else a / b - 1


def richardson(coarse, mid, fine, ratio=2.0):
    """Observed order p and extrapolated value from three solutions with refinement ratio 2."""
    if None in (coarse, mid, fine) or (mid - fine) == 0 or (coarse - mid) / (mid - fine) <= 0:
        return None, None
    p = np.log((coarse - mid) / (mid - fine)) / np.log(ratio)
    return float(p), float(fine + (fine - mid) / (ratio ** p - 1))


out = {}
# --- scaling --------------------------------------------------------------------------------------
ref = drift('scaling/ref')
out['scaling'] = {k: {'ratio': None if drift(f'scaling/{k}') is None or ref is None else drift(f'scaling/{k}') / ref,
                      'expected': e} for k, e in (('rho_half', 2.0), ('D_half', 2.0), ('rho_half_D_half', 4.0))}
for k, v in out['scaling'].items():
    v['error'] = None if v['ratio'] is None else v['ratio'] / v['expected'] - 1
    v['pass'] = v['error'] is not None and abs(v['error']) < 1e-9

# --- phase noise ----------------------------------------------------------------------------------
phase = [drift('convergence/T1_base')] + [drift(f'phase/T1_phi{p}') for p in (60, 120, 180, 240, 300)]
phase = [p for p in phase if p is not None]
lam = [drift(f'phase/T1_e0_lambda{l}') for l in (60, 150, 300)]
lam = [x for x in lam if x is not None]
noise = {}
if len(phase) >= 3:
    a = np.array(phase)
    noise['phi0'] = {'n': len(a), 'mean': float(a.mean()), 'std_rel': float(a.std(ddof=1) / abs(a.mean())),
                     'range_rel': float((a.max() - a.min()) / abs(a.mean()))}
if len(lam) >= 2:
    a = np.array(lam)
    noise['lambda_e0'] = {'n': len(a), 'mean': float(a.mean()), 'std_rel': float(a.std(ddof=1) / abs(a.mean())),
                          'range_rel': float((a.max() - a.min()) / abs(a.mean()))}
out['phase'] = noise
floor = noise.get('phi0', {}).get('std_rel', 0.0)

# --- convergence ----------------------------------------------------------------------------------
conv = {}
for T in ('T1', 'T2'):
    base = drift(f'convergence/{T}_base')
    row = {'base_m_s': base}
    for v in ('steps50', 'steps200', 'dz0.2', 'dz0.05', 'depth3', 'depth6', 'facets1280', 'orbits12'):
        row[v] = rel(drift(f'convergence/{T}_{v}'), base)
    p, ext = richardson(drift(f'convergence/{T}_steps50'), base, drift(f'convergence/{T}_steps200'))
    row['steps_order'], row['steps_error_of_base'] = p, rel(base, ext)
    p, ext = richardson(drift(f'convergence/{T}_dz0.2'), base, drift(f'convergence/{T}_dz0.05'))
    row['dz_order'], row['dz_error_of_base'] = p, rel(base, ext)
    o12 = load(f'convergence/{T}_orbits12')
    if o12:
        h = o12['result']['drift_history_m_s']
        row['orbit_history_rel_to_last'] = [x / h[-1] - 1 for x in h]
        tail = np.array(h[4:])
        row['orbit_scatter_after_4_rel'] = float(tail.std(ddof=1) / abs(tail.mean())) if len(tail) > 2 else None
        row['mean_orbits_5_12_vs_base'] = rel(float(tail.mean()), base) if len(tail) else None
    b = load(f'convergence/{T}_base')
    if b:
        row['base_orbits'], row['base_wall_s'] = b['result']['orbits'], b['wall_time_s']
        row['base_fallback_steps'] = b['result']['fallback_steps']
    tol = max(5e-3, 2 * floor)
    checks = [row.get(k) for k in ('steps200', 'dz0.05', 'depth6', 'facets1280')]
    row['accept_production_numerics'] = all(c is not None and abs(c) < tol for c in checks) if any(c is not None for c in checks) else None
    row['acceptance_tolerance'] = tol
    conv[T] = row
out['convergence'] = conv

# --- runtimes for the cost model -------------------------------------------------------------------
times = []
for p in RAW.rglob('provenance.json'):
    r = json.load(open(p))
    if not r.get('result'):
        continue
    inp = r.get('inputs', {})
    d = inp.get('tumbler_derived', {})
    cfg_path = Path(r['run_dir']) / 'config.yaml'
    import yaml
    cfg = yaml.safe_load(open(cfg_path))
    steps_orbit = int(np.ceil(365.25 * 24 * cfg['a_au'] ** 1.5 / d['P_dt_h'] * cfg['implicit_timesteps_per_day']))
    nf = 20 * 4 ** inp['numerics']['n_subdiv']
    times.append({'run': str(p.parent.relative_to(RAW)), 'wall_s': r['wall_time_s'], 'orbits': r['result']['orbits'],
                  'steps': steps_orbit * r['result']['orbits'], 'facets': nf, 'layers': cfg['n_layers']})
out['runtimes'] = times
if len(times) >= 4:
    A = np.array([[t['steps'], t['steps'] * t['facets'] * t['layers']] for t in times], float)
    y = np.array([t['wall_s'] for t in times])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    out['cost_model'] = {'per_step_s': float(coef[0]), 'per_cell_step_s': float(coef[1]),
                         'max_rel_residual': float(np.max(np.abs(A @ coef / y - 1)))}

json.dump(out, open(RAW.parent / 'summary.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != 'runtimes'}, indent=1, default=float))
