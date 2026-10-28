'''
Collect the Tempestovsky outputs of a batch (run_batch.py) and turn them into da/dt distributions.

    python collect.py inputs_sim.csv runs_sim [inputs_dig.csv runs_dig]

da/dt [au/Myr] = drift(Tempestovsky) * (D_SHAPE / D) * (RHO_SIM / rho) * roughness * shape_noise
'''
import csv, gzip, os, sys
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
        o = read_output(f)
        drift = float(o['drift (m/s)'])
        res.append(dict(run=int(r['run']), drift=drift, converged=o.get('converged') == 'True',
                        valid=o.get('valid', 'True') == 'True', orbits=int(o.get('orbits', 0)),
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
    sets = list(zip(sys.argv[1::2], sys.argv[2::2]))
    all_d = {}
    for inputs, run_dir in sets:
        res = collect(inputs, run_dir)
        all_d[run_dir] = summary(run_dir, res)
        np.save(os.path.join(run_dir, 'dadt_1e-4_au_Myr.npy'), all_d[run_dir])
