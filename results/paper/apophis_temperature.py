'''
Surface temperature of Apophis at perihelion for the poster figure (make_figures.py, fig_apophis_temperature).

Tempestovsky run of the nominal tumbling state of data/config/config_apophis.yaml (Pravec et al. 2014,
obliquity 167.2 deg, pre-flyby orbit) on the full radar shape. The orbit is repeated min_orbits = max_orbits
times. In the last orbit, among the timesteps within 2 deg of true anomaly from perihelion (+-1.6 days, longer
than one cycle of the tumbling seen from the Sun), the one is kept at which the Sun is 90 deg from the short
axis about H (azimuth about H, measured from the projection of the short axis, positive in the right-handed
sense about H; --delta). The figure shows the body from four sides 90 deg apart about H, and at this geometry
the short axis is seen obliquely in all of them. Stored are the surface temperatures of all facets, the body
orientation and the directions of the Sun and of the angular momentum.
Run from the repository root with the tempest environment:

    /home/dusan/anaconda3/envs/tempest/bin/python results/paper/apophis_temperature.py [--gamma 600] [--shape Apophis_3996.stl]

Writes results/paper/apophis_temperature.npz (the run itself happens in a temporary directory, so that the
files tempest.py writes into the working directory do not end up in the repository):
  vertices (n, 3, 3)  facet vertices in the body (principal-axis) frame [m]
  T (n,)              surface temperature [K]
  sun_body (3,)       unit vector to the Sun in the body frame
  H_body (3,)         unit vector of the rotational angular momentum in the body frame
  R_body_orbit (3,3)  body -> orbital frame (x towards perihelion, z along the orbital angular momentum)
  delta_deg           azimuth of the Sun about H from the short axis (target --delta)
  r_au, true_anomaly_deg, gamma, mean_phi_dot, mean_psi_dot [rad/s], drift (m/s, mean of the last orbit)
'''
import argparse, os, sys, tempfile
import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, 'results', 'paper', 'apophis_temperature.npz')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=600.0, help='thermal inertia [J m^-2 K^-1 s^-1/2]')
parser.add_argument('--shape', default='Apophis_3996.stl', help='shape model in data/shape_models')
parser.add_argument('--orbits', type=int, default=3, help='number of orbits (temperatures from the last one)')
parser.add_argument('--delta', type=float, default=90.0, help='azimuth of the Sun from the short axis about H [deg]')
args = parser.parse_args()
NU_WINDOW, DELTA = np.deg2rad(2.0), np.deg2rad(args.delta)

with open(os.path.join(REPO, 'data', 'config', 'config_apophis.yaml')) as f:
    cfg = yaml.safe_load(f)
work = tempfile.mkdtemp(prefix='apophis_temperature_')
cfg.update({'silent_mode': True, 'remote': False, 'shape_model_file': args.shape,
            'shape_model_directory': os.path.join(REPO, 'data', 'shape_models') + os.sep,
            'thermal_inertia': args.gamma, 'n_jobs': os.cpu_count(),
            'min_orbits': args.orbits, 'max_orbits': args.orbits,
            'output_file': os.path.join(work, 'yarko.txt'), 'progress_file': os.path.join(work, 'progress.txt')})
config_path = os.path.join(work, 'config.yaml')
with open(config_path, 'w') as f:
    yaml.safe_dump(cfg, f)
lam, bet = np.deg2rad(cfg['lambda_L']), np.deg2rad(cfg['beta_L'])
H_ORBIT = np.array([np.cos(bet) * np.cos(lam), np.cos(bet) * np.sin(lam), np.sin(bet)])   # in the orbital frame

import src.model.insolation as insolation_module
import src.model.solvers.tempestovsky as solver_module
from src.model.solvers.tempestovsky_implicit import YarkovskyImplicitSolver
from src.utilities.tumbling_final_2 import mean_euler_rates

# The solver loop is not changed; three wrappers follow it. Per orbit section the solver calls
# calculate_insolation_orbit_section (true anomaly, distance), which calls compute_tumbling_dynamics
# (rotation matrices, Sun in the body frame), and then orbit_step once per timestep of the section.
state = {'orbit': 0, 'k': 0, 'best': None}
orig_section = solver_module.calculate_insolation_orbit_section
orig_tumbling = insolation_module.compute_tumbling_dynamics
orig_step = YarkovskyImplicitSolver.orbit_step


def tumbling(*a, **kw):
    state['rotation'] = orig_tumbling(*a, **kw)
    return state['rotation']


def section(thermal_data, shape_model, simulation, config, steps, orbit_section, **kw):
    out = orig_section(thermal_data, shape_model, simulation, config, steps, orbit_section, **kw)
    state.update(measured=kw.get('initialisation') == 0, true_anomaly=out[1], distance=out[2], k=0,
                 shape_model=shape_model, simulation=simulation)
    if state['measured'] and orbit_section == 0:
        state['orbit'] += 1
    return out


def sun_azimuth(R, sun):
    """Azimuth of the Sun about H from the projection of the short axis (the half on the side of H)."""
    H = R.T @ H_ORBIT
    z = np.array([0, 0, np.sign(H[2])])
    z_perp, sun_perp = z - (z @ H) * H, sun - (sun @ H) * H
    return np.arctan2(H @ np.cross(z_perp, sun_perp), z_perp @ sun_perp)


def step(self, thermal_data, current_insolation, simulation):
    T = orig_step(self, thermal_data, current_insolation, simulation)
    k = state['k']
    state['k'] += 1
    nu = np.angle(np.exp(1j * state['true_anomaly'][k]))
    if state['measured'] and abs(nu) < NU_WINDOW:
        R = state['rotation']['rotations'][k]
        sun = state['rotation']['r_body'][k] / np.linalg.norm(state['rotation']['r_body'][k])
        miss = abs(np.angle(np.exp(1j * (sun_azimuth(R, sun) - DELTA))))
        best = state['best']
        if best is None or best['orbit'] < state['orbit'] or miss < best['miss']:
            state['best'] = {'orbit': state['orbit'], 'miss': miss, 'r': state['distance'][k], 'T': np.array(T[:, 0]),
                             'true_anomaly': nu, 'R': R.copy(), 'sun_body': sun}
    return T


insolation_module.compute_tumbling_dynamics = tumbling
solver_module.calculate_insolation_orbit_section = section
YarkovskyImplicitSolver.orbit_step = step

os.chdir(work)
sys.argv = ['tempest.py', '--config', config_path]
import tempest
tempest.main()

best, simulation = state['best'], state['simulation']
H_body = best['R'].T @ H_ORBIT
sun_body = best['sun_body']
mean_phi_dot, mean_psi_dot, _ = mean_euler_rates(simulation.y0, simulation.I)
drift = float(open(cfg['output_file']).readline().split(':')[1])
vertices = np.array([facet.vertices for facet in state['shape_model']])
np.savez_compressed(OUT, vertices=vertices, T=best['T'], sun_body=sun_body, H_body=H_body, R_body_orbit=best['R'],
         r_au=best['r'], true_anomaly_deg=np.rad2deg(best['true_anomaly']), gamma=args.gamma,
         mean_phi_dot=mean_phi_dot, mean_psi_dot=mean_psi_dot, drift=drift, shape=args.shape,
         delta_deg=np.rad2deg(sun_azimuth(best['R'], sun_body)))
print(f"orbit {best['orbit']}, r = {best['r']:.4f} au, true anomaly {np.rad2deg(best['true_anomaly']):.2f} deg, "
      f"Sun azimuth from the short axis {np.rad2deg(sun_azimuth(best['R'], sun_body)):.1f} deg, "
      f"T {best['T'].min():.1f}..{best['T'].max():.1f} K, angle(H, z_body) {np.rad2deg(np.arccos(H_body[2])):.1f} deg, "
      f"P_phi {2 * np.pi / mean_phi_dot / 3600:.2f} h, P_psi {2 * np.pi / mean_psi_dot / 3600:.2f} h, drift {drift:.4e} m/s")
print('written', OUT)
