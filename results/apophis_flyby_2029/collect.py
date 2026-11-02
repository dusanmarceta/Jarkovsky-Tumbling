'''
Collect the Tempestovsky outputs of a batch (run_batch.py) and turn them into da/dt distributions.

    python collect.py inputs_sim.csv.gz runs_sim [inputs_dig.csv.gz runs_dig] [--table drift_runs.csv.gz]

--table writes one row per finished run (both sets) to a small gzipped CSV, e.g. to commit the results
instead of the run folders.

da/dt [au/Myr] = drift(Tempestovsky) * (D_SHAPE / D) * (RHO_SIM / rho) * roughness * shape_noise
'''
import argparse, csv, gzip, os
import numpy as np

MS2AUMYR = 3.15576e13 / 1.495978707e11          # m/s -> au/Myr
D_SHAPE = 334.6                                   # volume-equivalent diameter of Apophis_200.stl [m]
RHO_SIM = 2000.0                                  # density in config_apophis.yaml [kg/m3]


def read_output(path):
    out = {}
    for line in open(path):
        k, _, v = line.partition(':')
        out[k.strip()] = v.strip()
    return out


def collect(inputs, run_dir):
    rows = list(csv.DictReader((gzip.open if inputs.endswith('.gz') else open)(inputs, 'rt')))
    res = []
    for r in rows:
        f = os.path.join(run_dir, f'run_{int(r["run"]):05d}.txt')
        if not os.path.exists(f):
            continue
        try:
            o = read_output(f)
            drift = float(o['drift (m/s)'])
        except (KeyError, ValueError):            # still being written
            continue
        T = o.get('T surface min, max last orbit (K)', 'nan nan').split()
        res.append(dict(run=int(r['run']), drift=drift, converged=o.get('converged') == 'True',
                        valid=o.get('valid', 'True') == 'True', orbits=int(o.get('orbits', 0)),
                        drift_last_orbit=float(o.get('drift last orbit (m/s)', 'nan')),
                        estimated_error=float(o.get('estimated error (m/s)', 'nan')),
                        fallback_steps=int(o.get('backward Euler fallback steps', 0)), T_min=float(T[0]), T_max=float(T[1]),
                        dadt=drift * MS2AUMYR * (D_SHAPE / float(r['D'])) * (RHO_SIM / float(r['rho']))
                        * float(r['roughness']) * float(r['shape_noise']),
                        **{k: float(r[k]) for k in ('thermal_inertia', 'albedo', 'lambda_L', 'beta_L', 'P_phi_h', 'E_ratio', 'sam')}))
    return res


def summary(name, res):
    d = np.array([x['dadt'] for x in res if x['valid']]) * 1e4
    bad = sum(not x['valid'] for x in res); nc = sum(not x['converged'] for x in res)
    p = np.percentile(d, [0.135, 15.865, 50, 84.135, 99.865])
    print(f'{name}: {len(res)} runs ({bad} invalid, {nc} not converged), mean orbits {np.mean([x["orbits"] for x in res]):.1f}')
    print(f'   da/dt [1e-4 au/Myr]: mean {d.mean():.2f}, sd {d.std():.2f}, median {p[2]:.2f}, '
          f'1 sigma [{p[1]:.2f}, {p[3]:.2f}], 3 sigma [{p[0]:.2f}, {p[4]:.2f}]')
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='+', help='inputs file and run folder, repeated for each set')
    ap.add_argument('--table', help='write all finished runs to this .csv.gz')
    a = ap.parse_args()
    table = []
    for inputs, run_dir in zip(a.pairs[0::2], a.pairs[1::2]):
        res = collect(inputs, run_dir)
        np.save(os.path.join(run_dir, 'dadt_1e-4_au_Myr.npy'), summary(run_dir, res))
        name = os.path.basename(os.path.normpath(run_dir))
        table += [dict(set=name, **x) for x in res]
    if a.table:
        cols = ['set', 'run', 'drift', 'drift_last_orbit', 'estimated_error', 'valid', 'converged', 'orbits',
                'fallback_steps', 'T_min', 'T_max', 'dadt']
        with gzip.open(a.table, 'wt', newline='') as f:
            w = csv.writer(f)
            w.writerow(['set', 'run', 'drift_m_s', 'drift_last_orbit_m_s', 'estimated_error_m_s', 'valid', 'converged',
                        'orbits', 'fallback_steps', 'T_surface_min_K', 'T_surface_max_K', 'dadt_au_Myr'])
            for x in table:
                w.writerow([x[c] if not isinstance(x[c], float) else f'{x[c]:.10g}' for c in cols])
        print(f'{len(table)} runs written to {a.table}')
