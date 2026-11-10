"""
Run one Tempestovsky simulation in its own working directory and record provenance.

Every run gets:
    <run_dir>/config.yaml      the exact configuration passed to tempest.py
    <run_dir>/log.txt          stdout + stderr of tempest.py
    <run_dir>/result.txt       tempest.py output_file (drift, convergence history, diagnostics)
    <run_dir>/progress.txt     tempest.py progress_file
    <run_dir>/provenance.json  inputs, code commit, environment, wall time, exit code, parsed result

tempest.py writes execution_time.txt, insolation_data/, temperature_data/ and output/ into its working
directory, so each run uses its own directory and absolute paths for the shape model and the outputs.
No solver file is modified by this module.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TEMPEST = REPO / 'tempest.py'
SHAPES = REPO / 'data' / 'shape_models'


def git_state():
    def run(*args):
        return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {'commit': run('rev-parse', 'HEAD'), 'branch': run('branch', '--show-current'),
            'dirty_solver_files': run('status', '--porcelain', '--', 'tempest.py', 'src')}


def load_template(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_result(path):
    out = {}
    if not Path(path).exists():
        return None
    for line in open(path):
        k, _, v = line.partition(':')
        out[k.strip()] = v.strip()
    def num(key):
        try:
            return float(out[key])
        except (KeyError, ValueError):
            return None
    history = [float(v) for k, v in out.items() if k.startswith('orbit ') and k[6:].isdigit()]
    tmin, tmax = (out.get('T surface min, max last orbit (K)', 'nan nan').split() + ['nan', 'nan'])[:2]
    return {'drift_m_s': num('drift (m/s)'), 'drift_last_orbit_m_s': num('drift last orbit (m/s)'),
            'drift_extrapolated_m_s': num('drift extrapolated (m/s)'), 'estimated_error_m_s': num('estimated error (m/s)'),
            'converged': out.get('converged') == 'True', 'valid': out.get('valid') == 'True',
            'orbits': int(num('orbits') or 0), 'fallback_steps': int(num('backward Euler fallback steps') or 0),
            'fallback_fraction': num('backward Euler fallback fraction'), 'T_surface_bound_K': num('T surface bound (K)'),
            'T_surface_min_K': float(tmin), 'T_surface_max_K': float(tmax), 'drift_history_m_s': history}


def native(v):
    """numpy scalars/arrays -> plain Python (yaml.safe_dump cannot represent numpy types)."""
    if hasattr(v, 'tolist'):
        return v.tolist()
    if isinstance(v, dict):
        return {k: native(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [native(x) for x in v]
    return v


def run_case(config, run_dir, n_threads=1, extra_provenance=None, timeout=None):
    """Write config (dict) into run_dir, run tempest.py there, return the provenance record."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = native(dict(config))
    shape_dir = Path(cfg.get('shape_model_directory', 'data/shape_models/'))
    cfg['shape_model_directory'] = str(shape_dir if shape_dir.is_absolute() else REPO / shape_dir) + '/'
    cfg.update({'silent_mode': True, 'remote': False,
                'n_jobs': int(n_threads), 'output_file': str(run_dir / 'result.txt'),
                'progress_file': str(run_dir / 'progress.txt'),
                'animate_final_day_temp_distribution': False, 'save_visible_phase_curve_data': False,
                'calculate_visible_phase_curve': False, 'calculate_thermal_phase_curve': False,
                'save_thermal_phase_curve_data': False, 'plot_insolation_curve': False})
    for f in ('result.txt', 'progress.txt'):
        (run_dir / f).unlink(missing_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    env = dict(os.environ, NUMBA_NUM_THREADS=str(n_threads), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
    t0 = time.time()
    with open(run_dir / 'log.txt', 'w') as log:
        try:
            proc = subprocess.run([sys.executable, str(TEMPEST), '--config', str(run_dir / 'config.yaml')],
                                  cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 'timeout'
    wall = time.time() - t0
    import numpy, scipy, numba, astropy
    record = {'run_dir': str(run_dir), 'exit_code': code, 'wall_time_s': round(wall, 2), 'n_threads': n_threads,
              'result': parse_result(run_dir / 'result.txt'), 'git': git_state(),
              'environment': {'python': sys.version.split()[0], 'numpy': numpy.__version__, 'scipy': scipy.__version__,
                              'numba': numba.__version__, 'astropy': astropy.__version__,
                              'machine': platform.platform(), 'processor': platform.processor()},
              'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if extra_provenance:
        record['inputs'] = extra_provenance
    with open(run_dir / 'provenance.json', 'w') as f:
        json.dump(record, f, indent=1, default=str)
    return record
