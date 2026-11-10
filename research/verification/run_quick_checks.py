"""
Short solver runs (minutes) that the campaign design depends on.

  benchmark   cost per time step vs facets x layers, principal-axis and tumbling (a = 0.2 au shortens the orbit;
              the cost per step depends on facets, layers and rotation integration, not on the orbit length)
  cp          specific heat c_p changed at fixed Gamma and rho: the 1D solver must give the same drift
              (tests that thermal properties enter only through Gamma)
  scaling     drift ~ 1/rho and ~ 1/D at fixed Gamma (completes the group stopped earlier)

    python research/verification/run_quick_checks.py --workers 4
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(RESEARCH / 'verification'))
from run_verification import T1, jobs_scaling, tumbler  # noqa: E402
from tools.runner import run_case  # noqa: E402

OUT = RESEARCH / 'runs' / 'verification' / 'raw'


def jobs_benchmark():
    out = []
    for kind, u in (('pa', 0.0), ('tum', 0.25)):
        for n_subdiv, layers in ((2, 300), (2, 1200), (2, 3000), (3, 1200)):
            if kind == 'tum' and (n_subdiv, layers) == (3, 1200):
                continue
            phys = dict(T1, u_exc=u, P_par_h=24.0, ecc=0.0)
            cfg, st = tumbler(phys, {'n_subdiv': n_subdiv, 'steps_per_rotation': 100, 'min_orbits': 2, 'max_orbits': 2},
                              {'a_au': 0.2})
            cfg['n_layers'] = layers
            cfg['yarkovsky_abs_tolerance'] = 0.0
            out.append((f'benchmark/{kind}_f{20 * 4 ** n_subdiv}_L{layers}', cfg, st))
    return out


def jobs_cp():
    out = []
    fast = {'n_subdiv': 1, 'steps_per_rotation': 50, 'n_seasonal_depths': 3.0, 'dz_over_ld': 0.2,
            'min_orbits': 2, 'max_orbits': 2}
    for name, cp in (('cp1000', 1000.0), ('cp600', 600.0), ('cp1200', 1200.0)):
        cfg, st = tumbler(dict(T1, P_par_h=24.0), fast, {'specific_heat_capacity': cp})
        cfg['yarkovsky_abs_tolerance'] = 0.0
        out.append((f'cp/{name}', cfg, st))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    jobs = jobs_benchmark() + jobs_cp() + [j for j in jobs_scaling() if not j[0].endswith('rho_half_D_half')]

    def go(job):
        name, cfg, st = job
        d = OUT / name
        if (d / 'provenance.json').exists() and json.load(open(d / 'provenance.json')).get('result'):
            return name, 'skipped'
        rec = run_case(cfg, d, 1, extra_provenance=st['state'])
        r = rec['result'] or {}
        return name, f"exit {rec['exit_code']} drift {r.get('drift_m_s')} orbits {r.get('orbits')} {rec['wall_time_s']} s"

    with ThreadPoolExecutor(a.workers) as ex:
        for name, status in ex.map(go, jobs):
            print(name, status, flush=True)
