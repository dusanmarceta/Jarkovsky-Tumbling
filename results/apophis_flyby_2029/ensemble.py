'''Full ensemble of Benson et al. (Sec. 3.1): 50 000 runs, uniform inertias, poles in the admissible region,
uniform attitude/omega phases. Saves pre- and post-flyby states to ensemble_50000.npz.'''
import numpy as np, time
from multiprocessing import Pool
from flyby import *

N = 50000
rng = np.random.default_rng(2029)

def id_for(pair):
    return initial_I_d(pair[0], pair[1], 263.0, 27.38)

if __name__ == '__main__':
    t0 = time.time()
    # inertia pairs, rejecting the dynamically non-viable ones (no SAM state with the nominal periods)
    I1s, I2s, Ids = [], [], []
    with Pool(6) as pool:
        while len(I1s) < N:
            m = int(1.1 * (N - len(I1s))) + 10
            a = rng.uniform(0.53, 0.73, m); b = rng.uniform(0.95, 0.974, m)
            ids = pool.map(id_for, list(zip(a.tolist(), b.tolist())), chunksize=200)
            for x, y, z in zip(a, b, ids):
                if z is not None:
                    I1s.append(x); I2s.append(y); Ids.append(z)
    I1, I2 = np.array(I1s[:N]), np.array(I2s[:N])
    print(f'inertias done {time.time()-t0:.0f} s', flush=True)
    poles = sample_poles(rng, N)
    q0, w0, Id0, ok = initial_states(I1, I2, poles, rng)
    assert ok.all()
    print(f'initial states done {time.time()-t0:.0f} s', flush=True)
    dt = 30.0
    g, rhat, n_steps, jd_ca, Rmin = trajectory(dt)
    qf, wf, acc = integrate(q0, w0, np.c_[I1, I2, np.ones(N)], g, rhat, dt, n_steps, n_steps // 2)
    print(f'integration done {time.time()-t0:.0f} s', flush=True)
    pre = final_state(q0, w0, I1, I2)
    post = final_state(qf, wf, I1, I2)
    out = dict(I1=I1, I2=I2, acc_ca_deg_h2=np.degrees(acc) * HOUR**2, jd_ca=jd_ca, R_ca_km=Rmin, q0=q0, w0=w0, qf=qf, wf=wf)
    out.update({'pre_' + k: v for k, v in pre.items()})
    out.update({'post_' + k: v for k, v in post.items()})
    np.savez('ensemble_50000.npz', **out)
    print(f'saved {time.time()-t0:.0f} s', flush=True)
