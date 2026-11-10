"""
fastspin.derived_fast must reproduce the solver's own rotation set-up (spinstate.derived, which builds
src/model/simulation.Simulation and calls src/utilities/tumbling_final_2.mean_euler_rates).

Acceptance: relative difference < 1e-8 in P_par, P_dt, T_omega, P_psi, P_lc and the nutation range, identical
mode, on SAM and LAM states from every stratum.
"""
import numpy as np
import pytest

from tools.campaign import STRATA, state_from_unit
from tools.fastspin import derived_fast
from tools.spinstate import derived

KEYS = ('P_par_h', 'P_dt_h', 'P_psi_h', 'T_omega_h', 'P_lc_h', 'theta_min_deg', 'theta_max_deg', 'theta0_deg')


@pytest.mark.parametrize('stratum', list(STRATA))
def test_fast_derivation_matches_solver_setup(stratum):
    rng = np.random.default_rng(hash(stratum) % 2 ** 32)
    for x in rng.random((4, 12)):
        st = state_from_unit(x, stratum)
        a = (st['I1'], st['I2'], st['E_ratio'], 10.0, st['psi_0'], st['phi_0'])
        f, s = derived_fast(*a), derived(*a)
        assert f['mode'] == s['mode']
        for k in KEYS:
            if np.isfinite(s[k]) and s[k] != 0:
                assert abs(f[k] / s[k] - 1) < 1e-8, (k, f[k], s[k])
            else:
                assert (not np.isfinite(f[k])) or f[k] == s[k]
