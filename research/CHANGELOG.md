# Research working copy: change record

Branch `research/paper1`; baseline commit `f113ce6` contains the repository exactly as received
(`research/provenance/baseline_sha256.txt` holds checksums of every file). **No solver file (`tempest.py`, `src/`) has
been modified.** All additions live under `research/` plus three `.gitignore` lines.

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

## Solver issues found (not fixed, awaiting review)

**S-1. Crank–Nicolson surface solve can return a non-converged, unphysical temperature that is flagged valid.**
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
- Current mitigation (no solver change): the generator raises steps per rotation until S ≤ 1. The research
  acceptance rule rejects any run with fallbacks or with T_max > 1.05 × the perihelion subsolar temperature.
- Proposed correction: see `research/proposals/S1_surface_newton.md` (needs approval before it is applied).

**S-3 (performance, not physics). `mean_euler_rates` integrates 1000 rotations on every call.**
In `src/utilities/tumbling_final_2.py` (l. 61–67) the return-to-start event is not terminal, so `solve_ivp` runs to
1000 rotations at rtol 1e-11 (about 640,000 Python right-hand-side calls, 2–4 s). The solver calls it once per
simulation, which is harmless next to run times of minutes to hours. It made manifest generation slow until
`fastspin` was added. A one-line fix (`back_to_start.terminal = 2`, then take the first event after 1e-3 t_unit) is
possible, but it is not applied.

**S-2 (performance, not physics). The uniform depth grid makes fast rotators expensive.** n_layers ∝ √(P_orb/P_dt).
A geometric grid (the unused `update_thermal_state_thickness` / `calculate_layer_thicknesses` stubs) would cut
ordinary-stratum cost several-fold. It is a solver change and is not applied.
