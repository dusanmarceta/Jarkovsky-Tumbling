'''Apophis heliocentric orbits before and after the 2029 encounter (JPL Horizons, J2000 ecliptic, osculating).'''
import numpy as np
PRE = dict(epoch='2029-03-01', a=0.9223392094665867, e=0.1912037991142842, i=3.341908719103002, node=203.8646269833756, peri=126.6970482848714)
POST = dict(epoch='2029-05-30', a=1.103117216087997, e=0.1890712403388476, i=2.221108844784937, node=203.5634084924480, peri=71.43235419258967)

def orbit_matrix(o):
    '''Columns: perihelion direction, in-plane normal, orbit normal (J2000 ecliptic).'''
    O, i, w = np.radians([o['node'], o['i'], o['peri']])
    Rz = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    Rx = lambda t: np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return Rz(O) @ Rx(i) @ Rz(w)

def ecliptic_to_orbital(lam, beta, o):
    '''Spin axis (lambda, beta) in J2000 ecliptic -> (lambda_L, beta_L) in the orbital frame of orbit o
    (x towards perihelion, z along the orbital angular momentum), as used by Tempestovsky.'''
    lam, beta = np.radians(lam), np.radians(beta)
    v = np.stack([np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), np.sin(beta)], axis=-1)
    u = v @ orbit_matrix(o)          # = (M^T v) for each row
    return np.degrees(np.arctan2(u[..., 1], u[..., 0])) % 360, np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1)))

def obliquity(lam, beta, o):
    return 90.0 - ecliptic_to_orbital(lam, beta, o)[1]

if __name__ == '__main__':
    print('pre-flyby nominal pole (250, -75) in the pre-flyby orbital frame:', ecliptic_to_orbital(250.0, -75.0, PRE))
    print('pre-flyby nominal pole (250, -75) in the post-flyby orbital frame:', ecliptic_to_orbital(250.0, -75.0, POST))
