'''Grid: full vs linearized Tempestovsky, a = 0.5 au circular, 80-facet sphere, fixed-axis rotation.'''
import os, re, json, time, subprocess, itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
PY = os.environ.get('TEMPEST_PYTHON', '/home/dusan/anaconda3/envs/tempest/bin/python')
BASE = open(os.path.join(REPO, 'data/config/config_vokrouhlicky_test.yaml')).read()

A_AU = 0.5
P_ORB_H = 365.25 * 24 * A_AU**1.5
GAMMAS = [0, 45, 90]
PERIODS = [4, 12, 24]
TIS = [200, 500, 2000]
MODES = ['full', 'lin']
DZ = 0.1            # layer thickness in diurnal skin depths
N_SEASONAL = 4.5    # domain depth in seasonal skin depths
N_PARALLEL = 3
N_THREADS = 2


def setv(text, key, value):
    pat = re.compile(rf'^{key}:.*$', re.M)
    line = f'{key}: {value}'
    return pat.sub(line, text) if pat.search(text) else text + f'\n{line}\n'


def make_config(name, gamma, P, TI, mode, steps):
    n_skin = N_SEASONAL * np.sqrt(P_ORB_H / P)
    c = BASE
    for k, v in [('thermal_inertia', TI), ('a_au', A_AU), ('ecc', 0.0), ('beta_L', 90 - gamma), ('lambda_L', 0),
                 ('P_phi_h', P / 2), ('E_ratio', 1.0), ('n_skin_depths', round(n_skin, 2)),
                 ('n_layers', int(np.ceil(n_skin / DZ))), ('implicit_timesteps_per_day', steps),
                 ('n_jobs', N_THREADS), ('orbital_initialisation', 0),
                 ('linearized_emission', 'true' if mode == 'lin' else 'false'),
                 ('yarkovsky_tolerance', 1.0e-3), ('yarkovsky_abs_tolerance', 1.0e-9),
                 ('min_orbits', 3), ('max_orbits', 15), ('deep_temperature_correction', 'true'),
                 ('output_file', f'"{HERE}/out/{name}.txt"'), ('progress_file', f'"{HERE}/out/{name}_progress.txt"')]:
        c = setv(c, k, v)
    path = f'{HERE}/cfg/{name}.yaml'
    open(path, 'w').write(c)
    return path


def parse(path):
    if not os.path.exists(path):
        return None
    d = {}
    for line in open(path):
        k, _, v = line.partition(':')
        d[k.strip()] = v.strip()
    return d


def run(job):
    gamma, P, TI, mode = job
    name = f'g{gamma}_P{P}_TI{TI}_{mode}'
    out = f'{HERE}/out/{name}.txt'
    if parse(out):
        return name, parse(out)
    # Crank-Nicolson can give negative temperatures at low TI with too few steps (the run then aborts)
    for steps in (200, 400, 800):
        cfg = make_config(name, gamma, P, TI, mode, steps)
        t0 = time.time()
        with open(f'{HERE}/log/{name}.log', 'w') as log:
            subprocess.run([PY, 'tempest.py', '--config', cfg], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        res = parse(out)
        print(f'{name} steps={steps} {time.time() - t0:.0f}s ->', res and res.get('drift (m/s)'), flush=True)
        if res:
            res['steps'] = steps
            res['time_s'] = round(time.time() - t0)
            with open(out, 'a') as f:
                f.write(f"steps: {steps}\ntime_s: {res['time_s']}\n")
            return name, res
    return name, None


if __name__ == '__main__':
    for d in ('cfg', 'out', 'log'):
        os.makedirs(f'{HERE}/{d}', exist_ok=True)
    jobs = list(itertools.product(GAMMAS, PERIODS, TIS, MODES))
    jobs.sort(key=lambda j: j[1])   # shortest rotation period = most timesteps first
    with ThreadPoolExecutor(N_PARALLEL) as ex:
        results = dict(ex.map(run, jobs))
    json.dump(results, open(f'{HERE}/results.json', 'w'), indent=1)
    print('ALL DONE', flush=True)
