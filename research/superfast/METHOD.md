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
4. **Seasonal problem.** Each facet column is driven along the Kepler orbit by the averaged absorbed flux minus ε σ D₄, where D₄ = ⟨T⁴⟩ − ⟨T⟩⁴ comes from step 5.
   - The column uses the full solver's domain depth, 4.5 seasonal skin depths, with Δz = 0.05 l_s.
   - The time step is P_orb/4000. The scheme is Crank–Nicolson, or backward Euler if the surface is stiff.
   - The deep-temperature correction is applied per orbit, as in the solver.
   - The output, at each orbital position, is the mean surface temperature T̄ and the seasonal heat flux into the ground, q_s.
5. **Fast problem at frozen positions.** This is the IAC 2026 frozen-Sun method, extended. It runs at N_loc positions spaced uniformly in true anomaly.
   - The Sun is frozen and the body follows its exact torque-free attitude in absolute time.
   - Each facet column starts at T̄ and carries q_s as a constant surface sink.
   - The column uses the full solver's layer thickness and time step, Crank–Nicolson and the solver's own column kernel. Its depth is 8 skin depths of the slowest rotational period.
   - Averaging windows are whole body periods, chosen so the precession phase closes to within 5°.
   - A window is converged when the local da/dt changes by less than 2×10⁻⁴ between windows and the column energy imbalance is below 2×10⁻⁴ of the absorbed flux. The deep-temperature correction is applied only while the column is still taking up heat.
6. **Coupling.** Steps 4 and 5 are repeated three times by default, and the last change is reported.
7. **Orbit average.** The local da/dt are combined with time weights r_k².

`mode='diurnal'` gives the IAC 2026 method: no seasonal memory, no sink, and never-lit facets held at T = 0.

## Assumptions, and where they fail

- **Separation of time scales.** Every rotational period, including the body period and the precession period, must be ≪ P_orb. All 256 super-fast campaign states satisfy this (at most 0.025 P_orb). Near-commensurate rotational beats longer than about 0.1 P_orb break it.
- **Non-resonant tumbling.** The insolation average over the torus (step 3) and the window averages assume φ and the body phase are incommensurate. Resonant states need windows that cover the beat.
- **Thin fast layer.** The fast layer must be much thinner than the seasonal layer, so that T̄ is uniform over the fast column. The depth ratio is about 8 √(P_slow/P_orb), which is at most 0.03 in the super-fast stratum.
- **1D conduction.** This is shared with the full solver. It fails where the seasonal skin depth is not small compared with the body (`outside_1d_validity`). For super-fast bodies a few metres across with high Γ, l_s reaches tens of cm, so those runs stay excluded.

## Files

- `kernels.py`
- `twoscale.py`
- `run.py` (one manifest line per task; results in `<root>/runs_twoscale/<run_id>/`)
- `validation/iac_compare.py`
- `validation/overlap.py`
