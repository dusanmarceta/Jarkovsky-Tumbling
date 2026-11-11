# Two-scale method for fast and super-fast tumbling asteroids

Branch `research/superfast-method`. It is built from `research/paper1-hpc-v2`. The frozen tag `hpc-freeze-v1`,
the frozen manifests and the full solver are unchanged.

## What it computes

It computes the orbit-averaged Yarkovsky drift da/dt for any campaign manifest line: tumbler (SAM or LAM) or
principal-axis twin, any of the 12 campaign inputs, any eccentricity. The drift includes the diurnal and the
seasonal contributions and uses the same physics as the full solver:
- 1D conduction per facet;
- an insulating bottom;
- ε σ T⁴ emission;
- Lambertian recoil;
- the solver's Gauss formula, with radial and transverse terms.

## Why a new method

The full solver resolves every rotation over whole orbits. For a period of seconds that means about 10⁸–10⁹ time steps per orbit, and its uniform depth grid needs about 10⁴–10⁵ layers. That puts super-fast runs at 10³–10⁵ core-h each. The two-scale method only resolves rotations at a few frozen orbital positions, and it solves the seasonal problem on a coarse grid.

## Steps

1. **Attitude** (`attitude.py`). This step uses the solver's initial state (`Simulation.y0`), right-hand side and tolerances. One body period is integrated with dense output. The state at any absolute time is that period plus whole-period increments of φ and ψ. It agrees with `compute_tumbling_dynamics` to 10⁻¹²–10⁻¹⁰, which `tests/test_superfast.py` checks.
2. **Orbit** (`orbit.py`). The solver's convention is used: time is counted from aphelion, and s = −(cos f, sin f, 0).
3. **Tumbling-averaged insolation.** For each facet, ⟨max(0, n·s)⟩ is tabulated against cos ζ, where ζ is the angle between the Sun and L. One body period is sampled, and the average over precession is analytic.
4. **Seasonal problem.** Each facet column is driven along the Kepler orbit by the averaged absorbed flux and emits ε σ κ T̄⁴, where κ = ⟨T⁴⟩/⟨T⟩⁴ ≥ 1 comes from step 5 (κ = 1 in the first pass).
   - κ(f) between the frozen positions is a periodic cubic spline.
   - This multiplicative closure has the same fixed point as the additive one, ε σ (T̄⁴ + ⟨T⁴⟩ − ⟨T⟩⁴). The additive form drove the seasonal temperature below 0 K at low thermal inertia (Γ = 24), where ε σ (⟨T⁴⟩ − ⟨T⟩⁴) reaches about 275 W m⁻². The multiplicative form cannot.
   - The column uses the full solver's domain depth, 4.5 seasonal skin depths.
   - Δz = 0.05 l_q, where l_q = √(κ_th/ḟ_q) and ḟ_q is the rate of true anomaly at perihelion. For a circular orbit this is 0.05 l_s. At e = 0.71 it is 3× finer, which the short perihelion heat pulse needs: with 0.05 l_s, halving Δz still changed da/dt by 2.7 %.
   - The time step is P_orb/4000. The scheme is Crank–Nicolson, or backward Euler if the surface is stiff.
   - The deep-temperature correction is applied per orbit, as in the solver.
   - The output, at each orbital position, is the mean surface temperature T̄ and the seasonal heat flux into the ground, q_s.
5. **Fast problem at frozen positions.** This is the IAC 2026 frozen-Sun method, extended. It runs at N_loc positions spaced uniformly in true anomaly.
   - By default N_loc = 8⌈3√((1+e)/(1−e))⌉: 24 for e = 0, 64 for e = 0.71, 72 for e = 0.8. At e = 0.71, 24 positions were 0.9 % from the converged value; 48 and 96 differ by 0.08 %.
   - The Sun is frozen and the body follows its exact torque-free attitude in absolute time.
   - Each facet column starts at T̄ and carries q_s as a constant surface sink.
   - The column uses the full solver's layer thickness and time step, Crank–Nicolson and the solver's own column kernel. Its depth is 8 skin depths of the slowest rotational period.
   - Averaging windows are whole body periods, chosen so the precession phase closes to within 5°.
   - Near-resonant motion gets longer windows. If that first window closes the phase to worse than 0.2° per body period, the shortest window up to 60,000 steps with at most 0.2° per period is used. Otherwise the window drifts through a slow beat, and neither the beat average nor the energy test is reached. Without this, the 55.6 s super-fast tumbler 00231 (3 periods, 4.4°) was 1.1 % off.
   - A window is converged when two conditions hold. First, the local da/dt changes by less than 2×10⁻⁴ between windows. Second, the net (area-weighted) energy imbalance is below 2×10⁻⁴ of the absorbed flux.
   - The per-facet imbalance must also be below 2×10⁻⁴, or have stopped falling (less than 20 % over 3 windows) while staying below 5×10⁻³. The rounding of the window to whole time steps and the residual precession-phase mismatch leave a floor of 10⁻⁴–3×10⁻³ that no number of windows removes. Tightening the energy tolerance fourfold changed da/dt by at most 4×10⁻⁵.
   - The deep-temperature correction is applied while the per-facet imbalance is above 2×10⁻⁴.
6. **Coupling.** Steps 4 and 5 are repeated at least 3 times, and up to 6 times until the orbit-averaged drift changes by less than 10⁻³ between passes. At e = 0.71 a fixed 3 passes still left a 1 % change in the last pass.
7. **Orbit average.** The local da/dt are combined with time weights r_k².

`mode='diurnal'` gives the IAC 2026 method: no seasonal memory, no sink, and never-lit facets held at T = 0.

## Assumptions, and where they fail

- **Frozen Sun (quasi-static).** The Sun must hardly move during one rotation, including at perihelion. The control parameter is the true-anomaly advance per dominant rotational period at perihelion, Δf_q = ḟ_q P.
  - It is ≤ 0.07° for every super-fast run and ≤ 1.4° for the deferred fast runs.
  - It is 1–15° for the ~20 h overlap states, where the comparison with the full solver is affordable. That is why the overlap test is the hardest case for this method, not the easiest.
- **Separation of time scales.** Every rotational period, including the body period and the precession period, must be ≪ P_orb. All 450 super-fast deferred runs satisfy this; only one has its slowest rotational period above 0.01 P_orb, with a maximum of 0.025 P_orb. Near-commensurate rotational beats longer than about 0.1 P_orb break it.
- **Non-resonant tumbling.** The insolation average over the torus (step 3) and the window averages assume φ and the body phase are incommensurate. Resonant states need windows that cover the beat.
- **Thin fast layer.** The fast layer must be much thinner than the seasonal layer, so that T̄ is uniform over the fast column. The ratio of the fast column depth to the perihelion seasonal skin depth l_q has a median of 0.03 in the super-fast stratum. It exceeds 0.3 in 4 super-fast and 41 deferred fast runs, and 1 in 1 and 7 of them.
- **1D conduction.** This is shared with the full solver. It fails where the seasonal skin depth is not small compared with the body (`outside_1d_validity`). For super-fast bodies a few metres across with high Γ, l_s reaches tens of cm, so those runs stay excluded.

## Validation (research/runs/superfast_validation/VALIDATION.json)

- **IAC 2026 reference.** In `mode='diurnal'` the new method is within 0.04 % (PA) and 0.05 % (NPA) of the reference's Richardson values (`validation/iac_compare.py`).
- **Overlap with the full solver** (`validation/overlap.py`): 8 frozen states, each a tumbler and its PA twin, with 1.5–90 h periods.
  - Where the Sun moves at most 3° per rotation at perihelion, the methods agree within 0.9 % (median 0.16 %). The tumbling effect agrees within 0.33 % of the larger drift.
  - At the 1.5 h state: 0.11 % (tumbler) and 0.20 % (twin).
  - At 7–16° per rotation (e 0.71–0.80, ~20 h): +1.0 to +1.7 %. Dividing the period by 4 (`scaled.jsonl`) reduced the error to 0.3–0.9 %.
- **Convergence** (`validation/convergence.py`). For the fast and super-fast states every refinement changes da/dt by less than 0.1 %. The exception is halving the fast-layer thickness (the full solver's own Δz), which changes it by 0.1–0.5 %.
- **Unsupported campaign runs** (`validation/superfast_demo.py`): 8 deferred states, from P_φ = 5.0 s and D = 3.0 m to the most expensive fast run the full solver cannot execute. Every run converged.
- **Cost** (`cost.py`): calibrated at 6.4 ns per column-layer step. All 602 deferred runs together come to about 1,800 core-h (M4).

## Files

- `kernels.py`
- `twoscale.py`
- `run.py` (one manifest line per task; results in `<root>/runs_twoscale/<run_id>/`)
- `cost.py` (scheduler cost estimates)
- `validation/iac_compare.py`
- `validation/overlap.py`
- `validation/convergence.py`
- `validation/superfast_demo.py`
- `validation/report.py`
- HPC: `research/hpc/run_task_twoscale.sh`, `prepare.py --twoscale-costs`, `make_slurm_arrays.py --runner run_task_twoscale.sh`, `campaign/waves.py write --wave deferred --twoscale`
