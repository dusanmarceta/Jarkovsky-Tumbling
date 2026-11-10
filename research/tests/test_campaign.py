"""
Campaign sampler constraints (research/tools/campaign.py).

Acceptance criteria, on 4,000 random unit points per stratum (sampling only, no rotation integration):
  inertia from homogeneous ellipsoids: 0 < I1 <= I2 <= 1 and I1 + I2 > 1 (no impossible body)
  excitation: 1 < E_ratio < 1/I1, SAM exactly for u < 0.5, never within the separatrix band
  psi_0 reachable: I3 * A(psi_0) >= E_ratio for every LAM state (the solver would raise ValueError otherwise)
  isotropic L: sin(beta_L) uniform (mean ~ 0), all ranges inside the declared bounds
and on a few full states: the solver's own Simulation accepts the tumbler and twin configurations.
"""
import math

import numpy as np
import pytest

from tools.campaign import ECC, GAMMA, RHO, STRATA, lines_for_unit, state_from_unit
from tools.manifest import full_config
from tools.spinstate import simulation_for


@pytest.mark.parametrize('stratum', list(STRATA))
def test_sampled_states_are_admissible(stratum):
    rng = np.random.default_rng(42)
    sb = []
    for x in rng.random((4000, 12)):
        s = state_from_unit(x, stratum)
        I1, I2, E = s['I1'], s['I2'], s['E_ratio']
        assert 0 < I1 <= I2 <= 1 and I1 + I2 > 1
        assert 1 < E < 1 / I1
        assert (E < 1 / I2) == (s['u_exc'] < 0.5)
        r_sam, r_lam = 1 / I2 - 1, 1 / I1 - 1 / I2
        assert abs(E - 1 / I2) > 0.004 * min(r_sam, r_lam)
        psi = math.radians(s['psi_0'])
        A = math.sin(psi) ** 2 / I1 + math.cos(psi) ** 2 / I2
        assert A >= E - 1e-9
        lo, hi = STRATA[stratum]['P_phi_h']
        assert lo <= s['P_phi_h'] <= hi
        lo, hi = STRATA[stratum]['D_m']
        assert lo <= s['D_eq_m'] <= hi
        assert RHO[0] <= s['density'] <= RHO[1] and GAMMA[0] <= s['thermal_inertia'] <= GAMMA[1]
        assert ECC[0] <= s['ecc'] <= ECC[1]
        sb.append(math.sin(math.radians(s['beta_L'])))
    assert abs(np.mean(sb)) < 0.05


@pytest.mark.parametrize('stratum', ['slow', 'ordinary'])
def test_full_lines_accepted_by_solver_setup(stratum):
    rng = np.random.default_rng(7)
    for x in rng.random((2, 12)):
        for ln in lines_for_unit(x, stratum, 't', twins=('P_par', 'P_phi', 'P_lc')):
            c = full_config(ln['config_overrides'])
            sim = simulation_for(c['I1'], c['I2'], c['E_ratio'], c['P_phi_h'], c['psi_0'], c['phi_0'])
            assert sim.delta_t > 0
            if ln['kind'] != 'tumbler':                     # twin spins at the requested period
                assert abs(sim.rotation_period_s / 3600 / ln['twin_period_h'] - 1) < 1e-9
