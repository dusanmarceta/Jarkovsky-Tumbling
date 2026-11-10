"""
Shape-inertia consistency and the synthetic-asteroid generator.

Acceptance criteria
  mesh        : faceted volume equals the requested D_eq to 1e-12; principal axes coincide with x, y, z
                (off-diagonal / diagonal < 1e-12); outward normals; inertia ratios ordered I1 <= I2 <= 1 and
                obey the triangle inequality
  solver view : the STL read back by the solver (tempest.read_shape_model) gives the same volume through
                tempestovsky.geometry_for_yarko (mass = rho * V) to 1e-9, and unit outward normals
  generator   : every sampled state satisfies the constraints, reproduces its target P_par and E_ratio, the
                solver's own Simulation accepts every generated configuration, run ids are unique, twins
                share the tumbler's time step and physical depth grid, and the same seed gives the same design
"""
import numpy as np
import pytest

from tools.ellipsoid import analytic_inertia_ratios, ellipsoid_mesh, inertia_summary, mesh_file
from tools.population import build_state, from_unit, full_config, sample_unit
from tools.runner import REPO


@pytest.mark.parametrize('b,c', [(0.95, 0.9), (0.8, 0.6), (0.45, 0.2)])
@pytest.mark.parametrize('n', [1, 2, 3])
def test_mesh_properties(b, c, n):
    tri = ellipsoid_mesh(b, c, 1000.0, n)
    s = inertia_summary(tri)
    assert abs(s['D_eq_m'] / 1000.0 - 1) < 1e-12
    assert s['max_offdiag_rel'] < 1e-12
    assert np.linalg.norm(s['com_m']) < 1e-9
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert np.all(np.einsum('ij,ij->i', nrm, tri.mean(1)) > 0)
    assert 0 < s['I1'] <= s['I2'] <= 1 and s['I1'] + s['I2'] >= 1
    a1, a2 = analytic_inertia_ratios(b, c)
    assert abs(s['I1'] - a1) < 1e-10 and abs(s['I2'] - a2) < 1e-10   # icosahedral symmetry: exact


def test_solver_reads_mesh_consistently(tmp_path):
    from tempest import read_shape_model
    from src.model.solvers.tempestovsky import geometry_for_yarko
    path, s = mesh_file(0.7, 0.5, 1000.0, 2, tmp_path)
    facets = read_shape_model(str(path), 10, 5, 1, False)
    V, normals, areas = geometry_for_yarko(facets)
    assert abs(V / s['volume_m3'] - 1) < 1e-9
    assert abs(areas.sum() / s['area_m2'] - 1) < 1e-9
    pos = np.array([f.position for f in facets])
    assert np.all(np.einsum('ij,ij->i', normals, pos) > 0)
    assert np.allclose(np.linalg.norm(np.array([f.normal for f in facets]), axis=1), 1, atol=1e-9)


@pytest.fixture(scope='module')
def design(tmp_path_factory):
    d = tmp_path_factory.mktemp('meshes')
    U = sample_unit(16, seed=12345)
    return [build_state(from_unit(u), d) for u in U], U


def test_design_reproducible():
    assert np.array_equal(sample_unit(16, seed=12345), sample_unit(16, seed=12345))
    assert not np.array_equal(sample_unit(16, seed=12345), sample_unit(16, seed=54321))


def test_generated_states_valid(design):
    from tools.spinstate import simulation_for
    states, _ = design
    ids = set()
    for st in states:
        s = st['state']
        I1, I2 = s['mesh']['I1'], s['mesh']['I2']
        assert 0 < I1 <= I2 <= 1 and I1 + I2 >= 1
        assert 1.0 <= s['E_ratio'] <= 1 / I1
        tum = [r for r in st['runs'] if r[1] in ('tumbler', 'pa')][0]
        for rid, kind, cfg, rec in st['runs']:
            assert rid not in ids
            ids.add(rid)
            full = full_config(cfg)
            sim = simulation_for(full['I1'], full['I2'], full['E_ratio'], full['P_phi_h'], full['psi_0'], full['phi_0'],
                                 extra={k: full[k] for k in ('thermal_inertia', 'density', 'specific_heat_capacity',
                                                             'n_skin_depths', 'n_layers', 'a_au', 'lambda_L', 'beta_L',
                                                             'implicit_timesteps_per_day')})
            # same time step and same physical layer thickness as the tumbler (within rounding of the step count)
            assert abs(sim.delta_t / s['tumbler_derived']['delta_t_s'] - 1) < 0.02 or kind in ('tumbler', 'pa')
            ref = full_config(tum[2])
            sim_ref = simulation_for(ref['I1'], ref['I2'], ref['E_ratio'], ref['P_phi_h'], ref['psi_0'], ref['phi_0'],
                                     extra={k: ref[k] for k in ('thermal_inertia', 'density', 'specific_heat_capacity',
                                                                'n_skin_depths', 'n_layers', 'a_au',
                                                                'implicit_timesteps_per_day')})
            assert abs(sim.layer_thickness / sim_ref.layer_thickness - 1) < 1e-9
            assert full['n_layers'] == ref['n_layers']
            if kind.startswith('pa'):
                assert full['E_ratio'] == 1.0
