"""
The principal-axis limit and the period conventions (stage-1 audit issue: P_phi_h near E_ratio = 1).

These tests document the solver's convention and check that the research converter removes the discontinuity.
Acceptance criteria
  pa_branch      : at E_ratio = 1 the solver spins about z with period P_phi_h * A(psi_0) * I3
  discontinuity  : with fixed P_phi_h the spin timescale jumps between E_ratio = 1 and 1 + 1e-4 when I1 != I2
                   (> 1 %), and does not jump when I1 = I2 (< 1e-6)  -> characterises the defect
  converter      : at fixed target P_par the converted configuration reproduces P_par to 1e-9 and is continuous
                   across E_ratio -> 1 (P_dt differs from P_par by < 1e-3 at E_ratio = 1 + 1e-6 * range)
"""
import numpy as np
import pytest

from conftest import I_ELL
from tools.spinstate import derived, e_ratio_from_u, p_phi_for_period


def test_pa_branch_period():
    I1, I2 = I_ELL
    for psi in (0.0, 45.0, 90.0):
        d = derived(I1, I2, 1.0, 10.0, psi_0=psi)
        A = np.sin(np.radians(psi)) ** 2 / I1 + np.cos(np.radians(psi)) ** 2 / I2
        assert d['mode'] == 'PA'
        assert abs(d['P_dt_h'] / (10.0 * A) - 1) < 1e-12
        assert abs(d['P_par_h'] / d['P_dt_h'] - 1) < 1e-12


def test_p_phi_convention_is_discontinuous_for_triaxial_body():
    I1, I2 = I_ELL
    jump = derived(I1, I2, 1.0 + 1e-4, 10.0)['P_par_h'] / derived(I1, I2, 1.0, 10.0)['P_par_h'] - 1
    assert abs(jump) > 0.01
    sym = derived(0.6, 0.6, 1.0 + 1e-4, 10.0)['P_par_h'] / derived(0.6, 0.6, 1.0, 10.0)['P_par_h'] - 1
    assert abs(sym) < 1e-4


@pytest.mark.parametrize('u', [1e-6, 1e-3, 0.2, 0.5, 0.9])
def test_converter_hits_target(u):
    I1, I2 = I_ELL
    E = e_ratio_from_u(u, I1, I2, e_min_frac=0.0) if u < 1e-5 else e_ratio_from_u(u, I1, I2)
    P = p_phi_for_period(I1, I2, E, 12.5, 'P_par')
    assert abs(derived(I1, I2, E, P)['P_par_h'] / 12.5 - 1) < 1e-9


def test_converter_continuous_at_pa_limit():
    I1, I2 = I_ELL
    E = 1.0 + 1e-6 * (1 / I1 - 1)
    d_t = derived(I1, I2, E, p_phi_for_period(I1, I2, E, 12.5, 'P_par'))
    d_pa = derived(I1, I2, 1.0, p_phi_for_period(I1, I2, 1.0, 12.5, 'P_par'))
    assert abs(d_t['P_dt_h'] / d_pa['P_dt_h'] - 1) < 1e-3
    assert abs(d_t['P_par_h'] / d_pa['P_par_h'] - 1) < 1e-9


def test_excitation_map_avoids_separatrix_and_pa_branch():
    I1, I2 = I_ELL
    us = np.linspace(1e-9, 1, 2001)
    E = np.array([e_ratio_from_u(u, I1, I2) for u in us])
    r_sam, r_lam = 1 / I2 - 1, 1 / I1 - 1 / I2
    assert np.all(np.diff(E) >= 0)
    assert E.min() > 1 + 0.5e-3 * r_sam
    assert np.min(np.abs(E - 1 / I2)) > 0.004 * min(r_sam, r_lam)
    assert E.max() < 1 / I1
    assert np.all((E < 1 / I2) == (us < 0.5))           # balanced: u < 0.5 is SAM, u > 0.5 is LAM
