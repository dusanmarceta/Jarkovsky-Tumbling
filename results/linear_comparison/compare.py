'''Compare grid results (full / linearized Tempestovsky) with the linear Yarkovsky model.'''
import os, sys, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'Vokrouhlicky model'))
from linearni_model_eccentric import yarko_eccentric
from run_grid import GAMMAS, PERIODS, TIS, A_AU, parse

R = 238.97541362458387   # volume-equivalent radius of the 80-facet sphere
rho, cp, eps, albedo = 500, 1000., 0.95, 0.

rows = []
print(f"{'gamma':>5} {'P[h]':>4} {'TI':>5} | {'linear':>11} {'full':>11} {'lin.Tmp':>11} | {'full/lin-1':>10} {'linT/lin-1':>10} | orbits f/l  time f/l [s]")
for g, P, TI in itertools.product(GAMMAS, PERIODS, TIS):
    k = TI**2 / (cp * rho)
    lin = yarko_eccentric(A_AU, 0.0, rho, k, cp, R, g, P, 1 - albedo, eps, 0)
    f = parse(f'{HERE}/out/g{g}_P{P}_TI{TI}_full.txt')
    l = parse(f'{HERE}/out/g{g}_P{P}_TI{TI}_lin.txt')
    df = float(f['drift (m/s)']) if f else np.nan
    dl = float(l['drift (m/s)']) if l else np.nan
    rel = lambda d: 100 * (d / lin - 1)
    info = lambda r, key: r.get(key, '-') if r else '-'
    print(f"{g:>5} {P:>4} {TI:>5} | {lin:11.4e} {df:11.4e} {dl:11.4e} | {rel(df):9.2f}% {rel(dl):9.2f}% | "
          f"{info(f, 'orbits')}/{info(l, 'orbits')}  {info(f, 'time_s')}/{info(l, 'time_s')}")
    rows.append((g, P, TI, lin, df, dl))
np.savetxt(f'{HERE}/comparison.txt', np.array(rows), header='gamma P_h TI linear full linearized (m/s)')
