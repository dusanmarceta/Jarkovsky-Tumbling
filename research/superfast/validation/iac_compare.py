"""
Step 1 of the validation: the new implementation in 'diurnal' mode against the IAC 2026 frozen-Sun reference at
its own test case (R_eq = 5 m oblate ellipsoid c/a = 0.8, 80 facets, P0 = 60 s, gamma_L = 30 deg, lambda_L = 45 deg,
circular orbit at 1 au, f = 0; PA and the theta = 30 deg symmetric top at equal L).

The reference integrates explicit FTCS with a full-layer surface node at r = kappa dt/dz^2 = 0.4998
(dt = P0/388); the new code uses the full solver's Crank-Nicolson column with a half-layer surface node at the same
dt and dz. The reference itself changes by 0.6 % between 100 and 200 layers (stage 4A.1), so agreement is judged
against both grids and against its Richardson value 2 D_200 - D_100.

    python -m superfast.validation.iac_compare --mesh <Tempestovsky-main>/data/shape_models/ellipsoid_Req5m_ca0p8_80.stl
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[2]
for p in (str(RESEARCH), str(RESEARCH.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from superfast.twoscale import Numerics, TwoScaleYarkovsky  # noqa: E402
from tools.manifest import full_config  # noqa: E402

# reference values (Tempestovsky-main/data/output/iac26_reduced): non-bracketed protocol, T0 = 250 K, 100 layers
# (reproduced bit for bit), and the bracketed stage 4A.1 values at 100 and 200 layers
REF = {'pa': {'100': 2.936982645098467e-04, '100_bracket': 2.9392316176621807e-04, '200_bracket': 2.921665939246441e-04},
       'npa': {'100': 2.4390304895185113e-04, '100_bracket': 2.4391487412726582e-04, '200_bracket': 2.4231153152861690e-04}}

Req, ca, rho, cp, k, eps, A, P0 = 5.0, 0.8, 2000.0, 800.0, 0.01, 0.95, 0.0, 60.0
I1 = (1 + ca ** 2) / 2                       # A/C of the oblate ellipsoid (a = b, c = ca a)
theta = math.radians(30.0)


def config(mode, mesh):
    E = 1.0 if mode == 'pa' else 1.0 + math.sin(theta) ** 2 * (1 / I1 - 1)
    P_phi_s = P0 * I1                        # equal L = C 2 pi / P0: phi_dot = L / A
    return full_config({'I1': I1, 'I2': I1, 'E_ratio': E, 'P_phi_h': P_phi_s / 3600, 'phi_0': 0.0, 'psi_0': 90.0,
                        'lambda_L': 45.0, 'beta_L': 60.0, 'thermal_inertia': math.sqrt(k * rho * cp), 'density': rho,
                        'specific_heat_capacity': cp, 'emissivity': eps, 'albedo': A, 'a_au': 1.0, 'ecc': 0.0,
                        'n_skin_depths': 18, 'n_layers': 100, 'implicit_timesteps_per_day': 388,
                        'shape_model_file': Path(mesh).name, 'shape_model_directory': str(Path(mesh).parent) + '/'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    skin = math.sqrt(k / (rho * cp * 2 * math.pi / P0))
    rows = []
    for mode in ('pa', 'npa'):
        for layers in (100, 200):
            num = Numerics(mode='diurnal', n_loc=1, dt_s=P0 / (388 * (layers // 100) ** 2), dz_m=18 * skin / layers,
                           n_fast_layers=layers, mass_kg=rho * 4 / 3 * math.pi * Req ** 3, max_windows=4000,
                           tol_rel=1e-5, tol_abs_frac=0.0, window_phi_tol_deg=6.0)
            t0 = time.perf_counter()
            m = TwoScaleYarkovsky(config(mode, a.mesh), a.mesh, num)
            r = m.run()
            d = r['per_location'][0]['dadt']
            rows.append({'mode': mode, 'layers': layers, 'dadt_m_s': d, 'windows': r['per_location'][0]['windows'],
                         'window': r['window'], 'steps': r['total_fast_steps'], 'wall_s': time.perf_counter() - t0})
            print(json.dumps(rows[-1]))
    for mode in ('pa', 'npa'):
        d100 = next(x['dadt_m_s'] for x in rows if x['mode'] == mode and x['layers'] == 100)
        d200 = next(x['dadt_m_s'] for x in rows if x['mode'] == mode and x['layers'] == 200)
        R = REF[mode]
        rich_ref = 2 * R['200_bracket'] - R['100_bracket']
        print(f"{mode}: new 100 layers {d100:.6e} (ref {R['100']:.6e}, rel {d100 / R['100'] - 1:+.2e}); "
              f"new 200 {d200:.6e} (ref {R['200_bracket']:.6e}, rel {d200 / R['200_bracket'] - 1:+.2e}); "
              f"new 100->200 change {d200 / d100 - 1:+.2e} (ref {R['200_bracket'] / R['100_bracket'] - 1:+.2e}); "
              f"reference Richardson {rich_ref:.6e}: new 200 rel {d200 / rich_ref - 1:+.2e}")
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1)
