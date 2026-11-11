"""
Collect the validation results into one JSON summary (research/runs/superfast_validation/VALIDATION.json):
IAC reference comparison, overlap comparison with the full solver (drift, tumbling effect, cost), convergence
studies and the super-fast demonstration.

    python -m superfast.validation.report
"""
import json
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[2]
for p in (str(RESEARCH), str(RESEARCH.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import math  # noqa: E402

import numpy as np  # noqa: E402

from tools.manifest import accepted, read_manifest  # noqa: E402

V = RESEARCH / 'runs' / 'superfast_validation'
GM_SUN, AU, G = 1.32712440018e20, 1.495978707e11, 6.674e-11
PRAVEC_C = 17.0      # tumbling damping time tau = P^3 / (C^3 D^2) Gyr (P in h, D in km): Pravec et al. (2005), Icarus 173


def diagnostics(ln):
    """Validity diagnostics of one manifest line (physical units; see METHOD.md, 'Assumptions')."""
    i, d, t = ln['inputs'], ln['derived'], ln['thermal']
    e = i['ecc']
    n = math.sqrt(GM_SUN / AU ** 3)                                   # a = 1 au for every campaign run
    fdot_q = n * (1 + e) ** 2 / (1 - e * e) ** 1.5
    if ln['kind'] == 'tumbler':
        P_dom = i['P_phi_h'] * 3600
        P_fast = min(x for x in (d['P_phi_h'], d.get('P_psi_h') or 1e99, d['T_omega_h']) if x and np.isfinite(x)) * 3600
        P_slow = d['slowest_rotational_period_h'] * 3600
    else:
        P_dom = P_fast = P_slow = ln['twin_period_h'] * 3600
    kap = t['kappa_m2_s']
    l_s = t['l_seasonal_m']
    l_q = l_s / math.sqrt((1 + e) ** 2 / (1 - e * e) ** 1.5)
    w = 2 * math.pi / P_fast
    out = {'sun_motion_per_rotation_q_deg': math.degrees(fdot_q * P_dom),
           'sun_motion_per_slowest_q_deg': math.degrees(fdot_q * P_slow),
           'P_slow_over_P_orb': P_slow * n / (2 * math.pi),
           'fast_layer_over_l_q': 8 * math.sqrt(kap * P_slow / (2 * math.pi)) / l_q,
           'l_s_over_D': l_s / i['D_eq_m'], 'l_d_fastest_um': math.sqrt(kap * P_fast / (2 * math.pi)) * 1e6,
           'centrifugal_over_gravity': w * w / (4 / 3 * math.pi * G * i['density']),
           'Gamma': i['thermal_inertia'], 'D_m': i['D_eq_m'], 'ecc': e, 'P_phi_s': i['P_phi_h'] * 3600,
           'P_dominant_s': P_dom, 'P_slowest_s': P_slow, 'mode': d['mode'] if ln['kind'] == 'tumbler' else 'PA'}
    if ln['kind'] == 'tumbler':
        out['damping_time_yr'] = i['P_phi_h'] ** 3 / (PRAVEC_C ** 3 * (i['D_eq_m'] / 1000) ** 2) * 1e9
    return out


def load_new(files):
    new = {}
    for f in ([files] if isinstance(files, (str, Path)) else files):
        if Path(f).exists():
            new.update(json.load(open(f)))
    return new


def overlap(manifest, root, new_file):
    lines = read_manifest(manifest)
    new = load_new(new_file)
    rows = {}
    for ln in lines:
        p = Path(root) / 'runs' / ln['run_id'] / 'provenance.json'
        full = json.load(open(p)) if p.exists() else None
        n = new.get(ln['run_id'])
        r = (full or {}).get('result') or {}
        row = {'run_id': ln['run_id'], 'kind': ln['kind'], 'note': ln.get('validation_note'), 'stratum': ln['stratum'],
               'mode': ln['derived']['mode'], 'P_phi_h': ln['inputs']['P_phi_h'], 'ecc': ln['inputs']['ecc'],
               'Gamma': ln['inputs']['thermal_inertia'], 'D_m': ln['inputs']['D_eq_m'],
               'full_cost_estimate_core_h': ln['policy']['cost_core_h_estimate'], 'diagnostics': diagnostics(ln),
               'full_dadt': r.get('drift_m_s'), 'full_accepted': accepted(full) if full else None,
               'full_wall_s': (full or {}).get('wall_time_s'), 'full_orbits': r.get('orbits'),
               'full_history': r.get('drift_history_m_s')}
        if n:
            pl = n.get('per_location') or []
            row.update(new_windows_median=float(np.median([p['windows'] for p in pl])) if pl else None,
                       new_windows_max=max(p['windows'] for p in pl) if pl else None,
                       new_energy_facet_max=max((p.get('energy_imbalance_facet') or 0) for p in pl) if pl else None,
                       new_coupling_change_rel=(abs(n['iterations'][-1] - n['iterations'][-2]) / abs(n['iterations'][-1])
                                                if len(n['iterations']) > 1 else None),
                       speedup_vs_cost_model=(ln['policy']['cost_core_h_estimate'] * 3600 / n['cpu_s'] if n.get('cpu_s') else None))
            row.update(new_dadt=n['dadt_m_s'], new_iterations=n['iterations'], new_cpu_s=n.get('cpu_s'),
                       new_wall_s=n['wall_s'], new_converged=n['all_converged'], new_fast_steps=n['fast_steps'],
                       new_n_loc=n['numerics']['n_loc'], new_periods=n.get('periods'), new_window=n.get('window'))
            if row['full_dadt']:
                row['rel_diff'] = n['dadt_m_s'] / row['full_dadt'] - 1
            if row.get('full_wall_s') and n.get('cpu_s'):
                row['speedup_core_time'] = row['full_wall_s'] / n['cpu_s']
        if row['full_history'] and len(row['full_history']) > 1:
            h = row['full_history']
            row['full_last_orbit_change_rel'] = abs(h[-1] - h[-2]) / abs(h[-1])
            # interannual modulation: mean and spread of the per-orbit drift after the first (spin-up) orbit
            hh = h[1:] if len(h) > 2 else h
            m = sum(hh) / len(hh)
            row['full_mean_after_first'] = m
            row['full_orbit_spread_rel'] = (max(hh) - min(hh)) / abs(m)
            if n:
                row['rel_diff_to_mean'] = n['dadt_m_s'] / m - 1
        rows.setdefault(ln['state_id'], {})[ln['kind']] = row
    states = []
    for sid, d in rows.items():
        t, p = d.get('tumbler', {}), d.get('pa_P_par', {})
        s = {'state_id': sid, 'tumbler': t, 'twin': p}
        if all(x.get(k) is not None for x in (t, p) for k in ('full_dadt', 'new_dadt')):
            s['delta_full'] = t['full_dadt'] - p['full_dadt']
            s['delta_new'] = t['new_dadt'] - p['new_dadt']
            s['delta_diff_over_twin'] = (s['delta_new'] - s['delta_full']) / abs(p['full_dadt'])
            s['delta_diff_over_delta'] = (s['delta_new'] - s['delta_full']) / abs(s['delta_full'])
            if t.get('full_mean_after_first') is not None and p.get('full_mean_after_first') is not None:
                s['delta_full_mean'] = t['full_mean_after_first'] - p['full_mean_after_first']
                s['delta_diff_over_twin_mean'] = (s['delta_new'] - s['delta_full_mean']) / abs(p['full_mean_after_first'])
        states.append(s)
    return states


def cost_calibration(files, manifests):
    """Fit the cost model of superfast/cost.py to the measured runs: core s per column-layer step and windows per
    frozen position and coupling iteration."""
    lines = {ln['run_id']: ln for m in manifests for ln in read_manifest(m)}
    rows = []
    for rid, n in load_new(files).items():
        if not n.get('cpu_s') or rid not in lines:
            continue
        cells = n['fast_steps'] * 320 * n['n_fast_layers']             # every campaign mesh has 320 facets
        it = len(n['iterations'])
        nl = n['numerics']['n_loc']
        win = n['fast_steps'] / (it * nl * n['window']['steps'])
        rows.append({'run_id': rid, 'cpu_s': n['cpu_s'], 'cells': cells, 'windows_per_loc_iter': win,
                     'iterations': it, 'core_s_per_cell': n['cpu_s'] / cells})
    c = np.array([r['core_s_per_cell'] for r in rows if r['cells'] > 1e10])
    w = np.array([r['windows_per_loc_iter'] for r in rows])
    return {'runs': rows, 'C_CELL_median': float(np.median(c)) if len(c) else None,
            'windows_median': float(np.median(w)) if len(w) else None, 'windows_p90': float(np.percentile(w, 90)) if len(w) else None,
            'iterations_median': float(np.median([r['iterations'] for r in rows])) if rows else None}


def deferred_summary():
    """Diagnostics and two-scale cost estimates of the 602 deferred runs."""
    path, cpath = V / 'deferred_twoscale.jsonl', V / 'twoscale_costs.json'
    if not path.exists():
        return None
    lines = read_manifest(path)
    costs = json.load(open(cpath))['runs'] if cpath.exists() else {}
    out = {}
    for stratum in ('superfast', 'fast'):
        L = [ln for ln in lines if ln['stratum'] == stratum]
        D = [diagnostics(ln) for ln in L]
        c = np.array([costs[ln['run_id']]['twoscale_core_h_estimate'] for ln in L if ln['run_id'] in costs])
        f = np.array([ln['policy']['cost_core_h_estimate'] for ln in L])
        tum = [x for x, ln in zip(D, L) if ln['kind'] == 'tumbler']
        out[stratum] = {
            'runs': len(L), 'tumblers': len(tum), 'modes': {m: sum(x['mode'] == m for x in D) for m in ('SAM', 'LAM', 'PA')},
            'P_phi_s_range': [min(x['P_phi_s'] for x in D), max(x['P_phi_s'] for x in D)],
            'D_m_range': [min(x['D_m'] for x in D), max(x['D_m'] for x in D)],
            'Gamma_range': [min(x['Gamma'] for x in D), max(x['Gamma'] for x in D)],
            'sun_motion_per_rotation_q_deg_max': max(x['sun_motion_per_rotation_q_deg'] for x in D),
            'n_sun_motion_gt_1deg': sum(x['sun_motion_per_rotation_q_deg'] > 1 for x in D),
            'n_fast_layer_over_l_q_gt_0.3': sum(x['fast_layer_over_l_q'] > 0.3 for x in D),
            'n_fast_layer_over_l_q_gt_1': sum(x['fast_layer_over_l_q'] > 1 for x in D),
            'n_P_slow_over_P_orb_gt_0.01': sum(x['P_slow_over_P_orb'] > 0.01 for x in D),
            'l_s_over_D_max': max(x['l_s_over_D'] for x in D),
            'n_regolith_skin_lt_100um': sum(x['Gamma'] < 200 and x['l_d_fastest_um'] < 100 for x in D),
            'n_regolith_skin_lt_30um': sum(x['Gamma'] < 200 and x['l_d_fastest_um'] < 30 for x in D),
            'n_skin_lt_10um': sum(x['l_d_fastest_um'] < 10 for x in D),
            'centrifugal_over_gravity_min': min(x['centrifugal_over_gravity'] for x in D),
            'n_unbound_regolith_like': sum(x['centrifugal_over_gravity'] > 1 and x['Gamma'] < 200 for x in D),
            'n_unbound': sum(x['centrifugal_over_gravity'] > 1 for x in D),
            'n_tumblers_damping_lt_1e3_yr': sum(x['damping_time_yr'] < 1e3 for x in tum),
            'n_tumblers_damping_lt_1e6_yr': sum(x['damping_time_yr'] < 1e6 for x in tum),
            'twoscale_core_h_total': float(c.sum()) if len(c) else None,
            'twoscale_core_h_median': float(np.median(c)) if len(c) else None,
            'twoscale_core_h_max': float(c.max()) if len(c) else None,
            'twoscale_bins': {b: int(((c > lo) & (c <= hi)).sum()) for b, lo, hi in
                              (('1h', 0, 1), ('6h', 1, 6), ('24h', 6, 24), ('96h', 24, 96), ('400h', 96, 400), ('over', 400, 1e99))},
            'full_core_h_total': float(f.sum())}
    return out


if __name__ == '__main__':
    out = {'iac_compare': json.load(open(V / 'iac_compare.json')) if (V / 'iac_compare.json').exists() else None,
           'overlap': overlap(V / 'overlap.jsonl', V / 'raw' / 'full', sorted(V.glob('new_final_*.json'))),
           'overlap_preliminary': overlap(V / 'overlap.jsonl', V / 'raw' / 'full', V / 'new_default.json'),
           'scaled': overlap(V / 'scaled.jsonl', V / 'raw' / 'full', V / 'scaled_new.json') if (V / 'scaled.jsonl').exists() else None,
           'convergence': load_new(sorted(V.glob('convergence_final*.json'))),
           'convergence_development': load_new([V / 'convergence_pre_closure.json', V / 'convergence_dev.json']),
           'superfast_demo': overlap(V / 'superfast_demo.jsonl', V / 'raw' / 'full', sorted(V.glob('superfast_demo_new_*.json'))),
           'cost_calibration': cost_calibration(sorted(V.glob('new_final_*.json')) + sorted(V.glob('superfast_demo_new_*.json'))
                                                + [V / 'scaled_new.json'], [V / 'overlap.jsonl', V / 'superfast_demo.jsonl', V / 'scaled.jsonl']),
           'deferred': deferred_summary()}
    json.dump(out, open(V / 'VALIDATION.json', 'w'), indent=1, default=float)
    for s in out['overlap'] + (out['scaled'] or []):
        t, p = s['tumbler'], s['twin']
        f = lambda x: 'n/a' if x is None else f'{x:+.2e}'
        print(f"{s['state_id']:28s} tumbler rel {f(t.get('rel_diff'))} (to orbit mean {f(t.get('rel_diff_to_mean'))}, "
              f"spread {f(t.get('full_orbit_spread_rel'))})  twin rel {f(p.get('rel_diff'))} (to mean "
              f"{f(p.get('rel_diff_to_mean'))}, spread {f(p.get('full_orbit_spread_rel'))})  "
              f"Delta diff/twin {f(s.get('delta_diff_over_twin'))}  speed-up {t.get('speedup_core_time') or 0:.0f}x / "
              f"{p.get('speedup_core_time') or 0:.0f}x")
