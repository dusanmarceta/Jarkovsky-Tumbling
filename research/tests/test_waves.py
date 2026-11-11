"""
Wave selection, exclusions and the V2 convergence manifest (research/campaign/waves.py, build_convergence.py,
research/hpc/prepare.py task_bins). No simulation is run.

Acceptance criteria:
  the frozen manifests match FREEZE_checksums.json and V2 matches manifests_v2/CHECKSUMS.json
  the exclusion list is exactly the outside_1d_validity rule applied to the frozen runs (43 in executable routes)
  every unique frozen run is in exactly one wave, with the counts of the adjustment plan
  each wave obeys its rule (stratum, eccentricity, route)
  V2 base variants are the production runs themselves (same run_id and solver configuration)
  a written wave manifest holds each run once with its hpc block; task binning skips excluded runs and bins
  'full_long' runs only when allowed; the deferred wave with hpc.route = 'twoscale' is binned by two-scale estimates
"""
import json

import pytest

from campaign import waves as W
from hpc.prepare import task_bins

EXPECTED = {'excluded': 107, '0': 78, '1': 4810, '2': 2000, '3': 88, 'deferred': 602, '1b': 106}


@pytest.fixture(scope='module')
def frozen():
    return W.load_frozen()


@pytest.fixture(scope='module')
def supplements():
    return W.load_supplements()


@pytest.fixture(scope='module')
def waves(frozen, supplements):
    return W.assign(frozen, supplements)


def test_exclusion_list_is_the_rule(frozen):
    rebuilt = W.build_exclusions(frozen)
    stored = json.load(open(W.EXCLUSIONS))
    assert stored == json.loads(json.dumps(rebuilt))
    assert stored['counts']['in_executable_routes'] == 43
    assert all('outside_1d_validity' in W.unique_runs(frozen)[r['run_id']][0]['policy']['flags'] for r in stored['runs'])


def test_every_frozen_run_in_exactly_one_wave(frozen, waves):
    ids = set(W.unique_runs(frozen))
    assert len(ids) == 7685
    seen = [rid for w in W.FROZEN_WAVES for rid in waves[w]]
    assert len(seen) == len(set(seen)) == len(ids) and set(seen) == ids
    assert {w: len(r) for w, r in waves.items()} == EXPECTED


def test_wave_rules(waves):
    for ln, _, route in waves['1'].values():
        assert ln['stratum'] in ('slow', 'ordinary') and ln['inputs']['ecc'] <= 0.6 and route in W.EXECUTABLE
    for ln, _, route in waves['2'].values():
        assert (ln['inputs']['ecc'] > 0.6 or ln['stratum'] == 'fast') and route in W.EXECUTABLE
    for ln, _, route in waves['3'].values():
        assert ln['stratum'] == 'ordinary' and route == 'full_long' and ln['inputs']['thermal_inertia'] < 14
    for w in ('0', '1', '2', '3', '1b'):
        assert not any(W.outside_1d(ln) for ln, _, _ in waves[w].values())
    assert all(ln['stratum'] != 'superfast' for w in ('0', '1', '2', '3') for ln, _, _ in waves[w].values())


def test_v2_base_variants_are_production_runs(frozen, supplements):
    prod = W.unique_runs(frozen)
    base = [ln for ln in supplements['V2_convergence'] if ln['convergence']['variant'] == 'base']
    assert len(base) == 14                       # 7 corners x (tumbler + P_par twin)
    for ln in base:
        assert ln['run_id'] in prod and prod[ln['run_id']][0]['config_overrides'] == ln['config_overrides']


def test_written_wave_manifest(waves, tmp_path):
    out = tmp_path / 'wave3.jsonl'
    assert W.write_wave(waves, '3', out) == 88
    lines = [json.loads(l) for l in open(out)]
    assert len({ln['run_id'] for ln in lines}) == 88
    assert all(ln['hpc']['wave'] == '3' and ln['hpc']['route'] == 'full_long' and ln['hpc']['sources'] for ln in lines)
    bins, skipped = task_bins(lines)
    assert sum(map(len, bins.values())) == 0 and skipped == {'full_long': 88}
    bins, skipped = task_bins(lines, allow_long=True)
    assert len(bins['400h']) == 88 and not skipped
    bins, skipped = task_bins(lines, allow_long=True, excluded={lines[0]['run_id']})
    assert skipped == {'excluded': 1} and len(bins['400h']) == 87


def test_deferred_wave_twoscale_route(waves, tmp_path):
    """The deferred wave written for the two-scale method keeps the frozen lines unchanged (only the hpc block) and
    is binned only with two-scale cost estimates."""
    out = tmp_path / 'deferred_twoscale.jsonl'
    n = W.write_wave(waves, 'deferred', out, 'twoscale')
    lines = [json.loads(l) for l in open(out)]
    assert n == len(lines) == EXPECTED['deferred']
    assert all(ln['hpc']['route'] == 'twoscale' for ln in lines)
    frozen_lines = {rid: ln for rid, (ln, _, _) in waves['deferred'].items()}
    assert all(json.dumps({k: v for k, v in ln.items() if k != 'hpc'}, sort_keys=True, default=float)
               == json.dumps(frozen_lines[ln['run_id']], sort_keys=True, default=float) for ln in lines)
    bins, skipped = task_bins(lines)
    assert sum(len(v) for v in bins.values()) == 0 and skipped == {'twoscale': len(lines)}
    costs = {ln['run_id']: {'twoscale_core_h_estimate': 0.5} for ln in lines}
    bins, skipped = task_bins(lines, twoscale_costs=costs)
    assert len(bins['1h']) == len(lines) and not skipped
