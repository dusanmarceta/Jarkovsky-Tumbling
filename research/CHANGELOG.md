# Research working copy: change record

Branch `research/paper1`; baseline commit `f113ce6` contains the repository exactly as received
(`research/provenance/baseline_sha256.txt` holds checksums of every file). **One reviewed solver change**: fix S-1
(commit `fc9b41d`, approved by the user), 27 lines in `src/model/solvers/tempestovsky_implicit.py` and
`src/model/solvers/tempestovsky.py`. All other additions live under `research/` plus a few `.gitignore` lines.

| Change | Files | Supporting tests / evidence |
|---|---|---|
| Pinned environment record | `research/environment/` | baseline reproductions below |
| Isolated run wrapper with provenance | `research/tools/runner.py` | baselines reproduce |
| Baseline reproductions | `research/baseline/` | flyby run 0: −7.578562356e-6 vs −7.578562334e-6 m/s (3e-9); linear_comparison: 4 cases within 3.5e-5 (4-digit reference) |
| Ellipsoid meshes, mesh inertia | `research/tools/ellipsoid.py` | `tests/test_shape_and_generator.py` |
| Spin-state converter via the solver's `Simulation` | `research/tools/spinstate.py` | `tests/test_rotation.py`, `tests/test_spin_convention.py` |
| Balanced SAM/LAM excitation map | `spinstate.e_ratio_from_u` | `test_excitation_map_avoids_separatrix_and_pa_branch` |
| 12-input campaign sampler, routes, stiffness-limited time step | `research/tools/campaign.py`, `research/campaign/` | `tests/test_campaign.py` |
| Manifests, per-job runner, retry and acceptance | `research/tools/manifest.py` | acceptance adds a physical temperature bound (issue S-1) |
| HPC preparation, array scripts, aggregation | `research/hpc/` | not yet run on the cluster |
| Exact-scaling and c_p checks, benchmark | `research/verification/run_quick_checks.py` | 1/ρ exact, 1/D to 4e-12, c_p invariance 4.7e-14 |
| Fast rotation-state derivation for manifests (same equations and tolerances as the solver, stops after one polhode period); 380× faster; every ~50th state cross-checked against the solver's `Simulation` | `research/tools/fastspin.py`, `campaign.derive_state` | `tests/test_fastspin.py` (< 1e-8), build cross-checks max 9.3e-10 |
| Resumable manifest build (skips finished stages); manifests A–E and V built, archived as `.jsonl.gz` with `SUMMARY.json` | `research/campaign/build_all.sh`, `research/campaign/manifests/` | 8,394 lines, 7,685 unique runs, 0 rejected. A–D were derived through the solver's `Simulation`; E (and later builds) through `fastspin` (`derived.derivation = fastspin-v1`) |

## Solver issues

**S-1 (FIXED in `fc9b41d`). Crank–Nicolson surface solve could return a non-converged, unphysical temperature
that was flagged valid.**
- Where: `src/model/solvers/tempestovsky_implicit.py`, `surface_temperature` (Newton, 50 iterations, no convergence or
  root-existence check) and `implicit_column_step` (backward-Euler fallback only if the result is negative).
- Mechanism: when the surface stiffness S = rad_coeff·T³ = (2π/N)/(Δz/l_d)/Θ is large, the explicit half of the
  emission term makes the right-hand side R negative. The CN equation lin·x + nl·x⁴ = R then has no real root.
  Newton wanders and may stop at a positive value, which is accepted. The temperature then runs away; the solver's
  `valid` flag (finite and ≥ 0) does not catch it.
- Evidence: `research/runs/verification/raw/benchmark/pa_f1280_L1200`. A night-side facet went from 793 K to 6617 K
  at the first orbit step; T reached 1.5e76 K, with 1804 fallback steps, and the run was still "valid". The same
  settings with 320 facets survived with 67–114 fallbacks. S ≈ 0.7–1.05 gave no fallbacks (flyby campaign, scaling
  checks).
- Correction applied: `research/proposals/S1_surface_newton.md`. When R < 0 the solve returns NaN and the step falls
  back to backward Euler. Newton now starts from an upper bound and checks convergence. `valid = False` when
  T_max > 1.05 × the perihelion subsolar temperature (not with linearized emission).
- Validation (`research/verification/run_s1_validation.py`, `compare_s1.py`, results in `research/runs/postfix`):
  - Baselines are unchanged: flyby run 0 and the 4 linear cases agree to ≤ 2e-16.
  - The divergent case is finite: drift within 0.29 % of the 320-facet run. It is flagged invalid, as are all
    S ≫ 1 benchmark runs (T_max ≈ 960–1000 K against the 880 K bound). The bound passes for every S ≤ 1 run.
- **What the fix does not change: accuracy at S > 1.** Error of the stiff tumbler against the N = 4000 reference:

  | Method | Steps per rotation | S | Error |
  |---|---|---|---|
  | Crank–Nicolson | 100 | ≈ 10 | −3.8 % |
  | Crank–Nicolson | 300 | ≈ 3 | −3.9 % |
  | Crank–Nicolson | 1000 | ≈ 1 | +1.1e-5 |
  | backward Euler (θ = 1) | 100 | | −1.85 % |
  | backward Euler | 300 | | −0.63 % |
  | backward Euler | 1000 | | −0.19 % |
  | backward Euler | 3000 | | −0.065 % |

  CN at S ≤ 1 is therefore both the most accurate and the cheapest option. The campaign rule S ≤ 1
  (`campaign.S_TARGET`) stays, and the existing manifests remain valid. The research acceptance rule (no fallbacks,
  temperature bound) also stays.

**S-3 (performance, not physics). `mean_euler_rates` integrates 1000 rotations on every call.**
In `src/utilities/tumbling_final_2.py` (l. 61–67) the return-to-start event is not terminal, so `solve_ivp` runs to
1000 rotations at rtol 1e-11 (about 640,000 Python right-hand-side calls, 2–4 s). The solver calls it once per
simulation, which is harmless next to run times of minutes to hours. It made manifest generation slow until
`fastspin` was added. A one-line fix (`back_to_start.terminal = 2`, then take the first event after 1e-3 t_unit) is
possible, but it is not applied.

**S-2 (performance, not physics). The uniform depth grid makes fast rotators expensive.** n_layers ∝ √(P_orb/P_dt).
A geometric grid (the unused `update_thermal_state_thickness` / `calculate_layer_thicknesses` stubs) would cut
ordinary-stratum cost several-fold. It is a solver change and is not applied.
