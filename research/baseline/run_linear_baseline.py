"""
Baseline B2: re-run four cases of results/linear_comparison/run_grid.py with the current code and compare
with the values printed in results/linear_comparison/results_2026-10-14.txt (4 significant digits) and with
the analytic linear theory in 'Vokrouhlicky model/linearni_model_eccentric.py'.

Settings are copied from run_grid.make_config (80-facet sphere, a = 0.5 au, e = 0, fixed-axis rotation,
200 steps per rotation, dz = 0.1 diurnal skin depth, 4.5 seasonal skin depths, deep-temperature correction).
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runner import REPO, load_template, run_case  # noqa: E402

A_AU, DZ, N_SEASONAL = 0.5, 0.1, 4.5
P_ORB_H = 365.25 * 24 * A_AU ** 1.5
# (gamma, P_h, TI, mode): reference values from results_2026-10-14.txt
REFERENCE = {(0, 24, 200, 'full'): 1.3646e-05, (0, 24, 200, 'lin'): 1.3518e-05,
             (90, 24, 200, 'full'): -1.0036e-06, (90, 24, 200, 'lin'): -7.1887e-07}
LINEAR_THEORY = {(0, 24, 200): 1.3329e-05, (90, 24, 200): -6.8839e-07}
REF_ORBITS = {(0, 24, 200, 'full'): 3, (0, 24, 200, 'lin'): 3, (90, 24, 200, 'full'): 4, (90, 24, 200, 'lin'): 5}


def config_for(gamma, P, TI, mode):
    c = load_template(REPO / 'data/config/config_vokrouhlicky_test.yaml')
    n_skin = N_SEASONAL * np.sqrt(P_ORB_H / P)
    c.update({'thermal_inertia': TI, 'a_au': A_AU, 'ecc': 0.0, 'beta_L': 90 - gamma, 'lambda_L': 0,
              'P_phi_h': P / 2, 'E_ratio': 1.0, 'n_skin_depths': round(float(n_skin), 2),
              'n_layers': int(np.ceil(n_skin / DZ)), 'implicit_timesteps_per_day': 200, 'orbital_initialisation': 0,
              'linearized_emission': mode == 'lin', 'yarkovsky_tolerance': 1.0e-3, 'yarkovsky_abs_tolerance': 1.0e-9,
              'min_orbits': 3, 'max_orbits': 15, 'deep_temperature_correction': True})
    return c


def linear_theory(gamma, P, TI):
    sys.path.insert(0, str(REPO / 'Vokrouhlicky model'))
    from linearni_model_eccentric import yarko_eccentric
    rho, C = 500.0, 1000.0                      # config_vokrouhlicky_test.yaml
    K = TI ** 2 / (rho * C)
    return yarko_eccentric(A_AU, 0.0, rho, K, C, 238.975, gamma, P, 1.0, 0.95, 0)


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'runs' / 'baseline' / 'raw' / 'linear_comparison'
    cases = list(REFERENCE)
    run = lambda k: run_case(config_for(*k), out / f'g{k[0]}_P{k[1]}_TI{k[2]}_{k[3]}', 2)
    # first case alone: fills the numba cache and the visible-facet cache before parallel runs share them
    recs = {cases[0]: run(cases[0])}
    with ThreadPoolExecutor(3) as ex:
        recs.update(zip(cases[1:], ex.map(run, cases[1:])))
    rows = []
    for k, rec in recs.items():
        r = rec['result'] or {}
        d = r.get('drift_m_s')
        lt = LINEAR_THEORY[k[:3]]
        rows.append({'case': f'gamma {k[0]}, P {k[1]} h, TI {k[2]}, {k[3]}', 'drift_m_s': d, 'reference_m_s': REFERENCE[k],
                     'rel_diff_vs_reference': None if d is None else d / REFERENCE[k] - 1,
                     'linear_theory_m_s(repo table)': lt, 'linear_theory_m_s(recomputed)': linear_theory(*k[:3]),
                     'rel_diff_vs_linear_theory': None if d is None else d / lt - 1,
                     'orbits': r.get('orbits'), 'reference_orbits': REF_ORBITS[k], 'converged': r.get('converged'),
                     'valid': r.get('valid'), 'wall_time_s': rec['wall_time_s'], 'exit_code': rec['exit_code']})
    summary = Path(__file__).resolve().parents[1] / 'runs' / 'baseline' / 'linear_comparison_summary.json'
    json.dump(rows, open(summary, 'w'), indent=1)
    for r in rows:
        print(json.dumps(r))
