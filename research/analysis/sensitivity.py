"""
Sensitivity analysis of aggregated campaign results (research/hpc/aggregate.py output).

Outputs analysed per state (tumbler row joined with its pa_P_par twin):
  Y_signed  drift / S_lin             signed drift in units of the linear-theory scale
  Y_log     log10 |drift|             with the sign treated separately
  Y_delta   (drift - drift_twin) / S_lin    tumbling effect, never singular near zero drift
S_lin = |diurnal(gamma = 0)| + |seasonal(gamma = 90 deg)| of the linear theory (repository 'Vokrouhlicky model',
circular orbit at a, the state's D, rho, Gamma, c_p, P_par), a scale that never vanishes.

    python research/analysis/sensitivity.py morris --results results.csv --manifest A_morris_slow.jsonl
    python research/analysis/sensitivity.py saltelli --results results.csv --manifest F_saltelli_slow.jsonl
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
from tools.runner import REPO  # noqa: E402

NAMES = ['lambda_L', 'sin_beta_L', 'b_over_a', 'c_over_b', 'u_exc', 'log_P_phi', 'phi_0', 'psi_0',
         'density', 'log_Gamma', 'ecc', 'log_D']


def linear_scale(D, rho, Gamma, cp, P_h, a_au=1.0, alpha=0.9, eps=0.9):
    sys.path.insert(0, str(REPO / 'Vokrouhlicky model'))
    from linearni_model_eccentric import yarko_diurnal_circular, yarko_seasonal_circular
    K = Gamma ** 2 / (rho * cp)
    d = yarko_diurnal_circular(rho, K, cp, D / 2, a_au, 0.0, P_h, alpha, eps)
    s = yarko_seasonal_circular(rho, K, cp, D / 2, a_au, 90.0, P_h, alpha, eps)
    return abs(d) + abs(s)


def state_table(results, manifest):
    man = [json.loads(l) for l in open(manifest)]
    res = pd.read_csv(results)
    res = res[res['status'] == 'accepted'].set_index('run_id')
    rows = []
    for ln in man:
        if ln['kind'] != 'tumbler' or ln['run_id'] not in res.index:
            continue
        twin = next((m for m in man if m['state_id'] == ln['state_id'] and m['kind'] == 'pa_P_par'), None)
        d_t = res.loc[ln['run_id'], 'result.drift_m_s']
        d_p = res.loc[twin['run_id'], 'result.drift_m_s'] if twin and twin['run_id'] in res.index else np.nan
        i = ln['inputs']
        S = linear_scale(i['D_eq_m'], i['density'], i['thermal_inertia'], ln['fixed']['specific_heat_capacity'],
                         ln['derived']['P_par_h'])
        rows.append({'state_id': ln['state_id'], **dict(zip(NAMES, ln['unit'])), 'S_lin': S, 'drift': d_t, 'drift_twin': d_p,
                     'Y_signed': d_t / S, 'Y_log': math.log10(abs(d_t)) if d_t else np.nan, 'Y_delta': (d_t - d_p) / S})
    return pd.DataFrame(rows)


def morris(df, manifest_order, k=12, n_boot=1000, seed=0):
    """Elementary effects from consecutive points of each (k+1)-point trajectory (design_morris ordering)."""
    rng = np.random.default_rng(seed)
    X = df.set_index('state_id').reindex(manifest_order)
    out = {}
    for y in ('Y_signed', 'Y_log', 'Y_delta'):
        ee = {n: [] for n in NAMES}
        for t in range(len(manifest_order) // (k + 1)):
            seg = X.iloc[t * (k + 1):(t + 1) * (k + 1)]
            u = seg[NAMES].to_numpy()
            v = seg[y].to_numpy()
            for j in range(k):
                dx = u[j + 1] - u[j]
                m = int(np.argmax(np.abs(dx)))
                if np.isfinite(v[j]) and np.isfinite(v[j + 1]) and abs(dx[m]) > 0:
                    ee[NAMES[m]].append((v[j + 1] - v[j]) / dx[m])
        res = {}
        for n, e in ee.items():
            e = np.array(e)
            if len(e) < 3:
                res[n] = None
                continue
            boot = [np.mean(np.abs(rng.choice(e, len(e)))) for _ in range(n_boot)]
            res[n] = {'mu_star': float(np.mean(np.abs(e))), 'mu': float(np.mean(e)), 'sigma': float(np.std(e, ddof=1)),
                      'mu_star_ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], 'n': len(e)}
        out[y] = res
    return out


def saltelli(df, manifest_order, k=12):
    """First-order (Saltelli 2010) and total (Jansen 1999) indices from blocks A, B, A_B^(i) of design_saltelli."""
    X = df.set_index('state_id').reindex(manifest_order)
    n = len(manifest_order) // (k + 2)
    out = {}
    for y in ('Y_signed', 'Y_log', 'Y_delta'):
        v = X[y].to_numpy()
        fA, fB = v[:n], v[n:2 * n]
        ok = np.isfinite(fA) & np.isfinite(fB)
        var = np.var(np.concatenate([fA[ok], fB[ok]]))
        res = {}
        for i, name in enumerate(NAMES):
            fAB = v[(2 + i) * n:(3 + i) * n]
            m = ok & np.isfinite(fAB)
            res[name] = {'S1': float(np.mean(fB[m] * (fAB[m] - fA[m])) / var),
                         'ST': float(0.5 * np.mean((fA[m] - fAB[m]) ** 2) / var), 'n': int(m.sum())}
        out[y] = res
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('method', choices=['morris', 'saltelli'])
    ap.add_argument('--results', required=True)
    ap.add_argument('--manifest', required=True)
    a = ap.parse_args()
    order = [json.loads(l)['state_id'] for l in open(a.manifest) if json.loads(l)['kind'] == 'tumbler']
    df = state_table(a.results, a.manifest)
    print(json.dumps((morris if a.method == 'morris' else saltelli)(df, order), indent=1))
