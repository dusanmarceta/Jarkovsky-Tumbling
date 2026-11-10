"""
Scientific pilot: contrasting tumbling states, each with principal-axis (PA) twins at three period conventions.

States vary one factor at a time around a centre state C (plus a few combinations), covering obliquity,
elongation, excitation (SAM and LAM), thermal inertia, spin timescale and eccentricity with lambda_L.
Twins: P_par = 2*pi*L/(2E) (conserved spin rate about L), P_phi (mean precession period, convention of
Ďurech et al. 2026), P_lc = (1/P_phi - 1/P_psi)^-1 (main SAM light-curve period; undefined in LAM).

    python research/pilot/run_pilot.py build            # write manifest only
    python research/pilot/run_pilot.py run --workers 8  # run everything not yet accepted
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
from tools.manifest import lines_for_state, read_manifest, run_line, write_manifest  # noqa: E402

ROOT = RESEARCH / 'runs' / 'pilot'
C = {'cos_gamma': 0.70710678, 'lambda_L': 60.0, 'b_over_a': 0.8, 'c_over_b': 0.75, 'u_exc': 0.25,
     'P_par_h': 12.0, 'Gamma': 300.0, 'ecc': 0.2}
STATES = {
    'C_centre': {},
    'O90_obliquity90': {'cos_gamma': 0.0},
    'O150_retrograde': {'cos_gamma': -0.8660254},
    'O70_near_zero': {'cos_gamma': 0.3420201},
    'X05_near_PA': {'u_exc': 0.05},
    'X45_near_separatrix': {'u_exc': 0.45},
    'X70_LAM': {'u_exc': 0.70},
    'X90_deep_LAM': {'u_exc': 0.90},
    'G30_low_inertia': {'Gamma': 30.0},
    'G2000_high_inertia': {'Gamma': 2000.0},
    'S_elongated': {'b_over_a': 0.5, 'c_over_b': 0.6},
    'S_nearly_oblate': {'b_over_a': 0.92, 'c_over_b': 0.6},
    'P4_fast': {'P_par_h': 4.0},
    'P40_slow': {'P_par_h': 40.0},
    'E45_lambda0': {'ecc': 0.45, 'lambda_L': 0.0},
    'E45_lambda90': {'ecc': 0.45, 'lambda_L': 90.0},
    'K_LAM_obl90_G100_e03': {'u_exc': 0.8, 'cos_gamma': 0.0, 'Gamma': 100.0, 'ecc': 0.3},
}


def build(numerics=None):
    lines = []
    for sid, change in STATES.items():
        lines += lines_for_state(sid, None, dict(C, **change), ROOT / 'meshes', numerics=numerics)
    write_manifest(lines, ROOT / 'manifest.jsonl')
    return lines


def cost(line):
    rn, d = line['run_numerics'], line['derived']
    steps = 365.25 * 24 / rn['period_h'] * rn['steps_per_rotation']
    return steps * (0.3e-3 + 4.5e-9 * 20 * 4 ** line['numerics']['n_subdiv'] * rn['n_layers'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'run'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--numerics', default='{}', help='JSON overrides of population.NUMERICS')
    a = ap.parse_args()
    if a.cmd == 'build' or not (ROOT / 'manifest.jsonl').exists():
        lines = build(json.loads(a.numerics))
        print(f'{len(lines)} runs, estimated {sum(cost(l) for l in lines) * 5 / 3600:.1f} CPU-h (5 orbits each)')
    if a.cmd == 'run':
        lines = sorted(read_manifest(ROOT / 'manifest.jsonl'), key=lambda l: -cost(l))
        with ThreadPoolExecutor(a.workers) as ex:
            for line, rec in zip(lines, ex.map(lambda l: run_line(l, ROOT), lines)):
                r = rec.get('result') or {}
                print(line['state_id'], line['kind'], rec.get('exit_code'), r.get('drift_m_s'), r.get('orbits'),
                      r.get('valid'), r.get('converged'), rec.get('wall_time_s'), flush=True)
