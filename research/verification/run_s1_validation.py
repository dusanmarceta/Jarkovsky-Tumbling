"""
Validation of solver fix S-1 (research/proposals/S1_surface_newton.md). Writes to research/runs/postfix/.

  baselines   flyby campaign run 0 and the 4 linear_comparison cases must equal the pre-fix results to 1e-10
  benchmark   the 7 benchmark cases (a = 0.2 au, S ~ 10-40) must all be valid, finite and below the
              temperature bound; the formerly divergent pa_f1280_L1200 must agree with pa_f320_L1200 to ~mesh level
  stiffness   one stiff tumbler (benchmark tum_f320_L300) at 100, 300, 1000, 4000 steps per rotation: drift error
              relative to the 4000-step run as a function of S, and the fallback fraction
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(RESEARCH / 'verification'))
sys.path.insert(0, str(RESEARCH / 'baseline'))
from run_linear_baseline import REFERENCE, config_for  # noqa: E402
from run_quick_checks import jobs_benchmark  # noqa: E402
from tools.runner import REPO, run_case  # noqa: E402

OUT = RESEARCH / 'runs' / 'postfix'


def flyby():
    out = OUT / 'flyby_sim'
    subprocess.run([sys.executable, 'results/apophis_flyby_2029/run_batch.py', 'results/apophis_flyby_2029/inputs_sim.csv.gz',
                    '--out', str(out), '--last', '1'], cwd=REPO, check=True, capture_output=True)
    subprocess.run(['git', 'checkout', '--', 'execution_time.txt'], cwd=REPO, capture_output=True)
    from tools.runner import parse_result
    return parse_result(out / 'run_00000.txt')


def stiffness_jobs():
    jobs = []
    base = [j for j in jobs_benchmark() if j[0].endswith('tum_f320_L300')][0]
    for N in (100, 300, 1000, 4000):
        cfg = dict(base[1], implicit_timesteps_per_day=N)
        jobs.append((f'stiffness/tum_f320_L300_N{N}', cfg, base[2]))
    return jobs


if __name__ == '__main__':
    jobs = [(f'linear/g{k[0]}_P{k[1]}_TI{k[2]}_{k[3]}', config_for(*k), None) for k in REFERENCE]
    jobs += [(n.replace('benchmark/', 'benchmark/'), c, s) for n, c, s in jobs_benchmark()] + stiffness_jobs()

    def go(job):
        name, cfg, st = job
        rec = run_case(cfg, OUT / name, 1, extra_provenance=st['state'] if st else None)
        return name, rec

    first = go(jobs[0])                                   # recompiles the changed numba functions once
    with ThreadPoolExecutor(4) as ex:
        recs = dict([first] + list(ex.map(go, jobs[1:])))
    fb = flyby()
    summary = {'flyby_run0': fb, 'runs': {n: {'exit': r['exit_code'], 'wall_s': r['wall_time_s'], **(r['result'] or {})}
                                          for n, r in recs.items()}}
    json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1, default=float)
    print('done')
