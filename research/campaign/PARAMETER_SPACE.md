# Campaign parameter space (12 inputs, all eight rotational keys varied)

Implementation: `research/tools/campaign.py` (sampler, derived quantities, routes), `research/campaign/build_campaign.py`
(designs, manifests), `research/campaign/build_all.sh` (all manifests from fixed seeds). Constraint tests:
`research/tests/test_campaign.py`. Evidence tags: **[read]** full text read; **[checked]** confirmed online
(abstract or search result); **[verify]** check before citing; **[measured]** our own runs.

## A. The 12 inputs

| # | Code key | Physical meaning | Range | Sampling | Unit | Justification |
|---|---|---|---|---|---|---|
| 1 | `lambda_L` | longitude of angular momentum **L** in the orbit frame (from perihelion) | 0–360 | uniform | deg | no preferred apsidal orientation; acts only when e > 0 |
| 2 | `beta_L` | latitude of **L**; obliquity γ = 90° − β_L | −90–90 | sin β_L uniform (isotropic **L**) | deg | mathematical domain for the sensitivity analysis; NEA retrograde excess (La Spina et al. 2004 [verify]) is a population prior, not a domain limit |
| 3 | `I1` | I_x / I_z (long axis) | from shape, see C | via b/a ∈ [0.30, 0.98] | – | b/a distributions peak ~0.63–0.83 (Cibulková et al. 2016, arXiv:1610.02790 [checked]; Cibulková et al. 2018, A&A 611 A86 [checked]; McNeill et al. 2016 b/a = 0.83 ± 0.12 [checked]); fast rotators a/b ≥ 1.1–7.5 (Alarcon et al. 2026 [read]); 2008 TC3 1 : 0.54 : 0.36 (Scheirich et al. 2010 [checked]) |
| 4 | `I2` | I_y / I_z (intermediate axis) | from shape | via c/b ∈ [0.40, 0.98] | – | same sources; 2008 TC3 gives c/b = 0.67 |
| 5 | `E_ratio` | 2 I₃ E / L² (excitation) | 1 < E < 1/I1 | balanced u: (0, 0.5) SAM, (0.5, 1) LAM, uniform | – | no empirical distribution of excitation exists; Pravec et al. (2005) note a bias against detecting low-amplitude tumblers [checked] |
| 6 | `P_phi_h` | mean precession period (the observed tumbler period) | by stratum, see B | log-uniform | h | strata below |
| 7 | `phi_0` | initial Euler angle φ (attitude about **L**) | 0–360 | uniform | deg | phase; effect measured (study D) |
| 8 | `psi_0` | initial Euler angle ψ (position on the polhode) | reachable set | uniform over sin²ψ₀ ≥ (E − 1/I2)/(1/I1 − 1/I2) | deg | phase; SAM: full circle, LAM: two arcs |
| 9 | `density` | bulk density (mass) | 1000–3500 | uniform | kg m⁻³ | Marčeta et al. (2025) Table 2 [read]; Carry (2012) [verify] |
| 10 | `thermal_inertia` | Γ = √(Kρc_p) | 5–2500 | log-uniform | J m⁻² K⁻¹ s⁻½ | Marčeta et al. (2025) ranges K = 10⁻⁵–1 W m⁻¹ K⁻¹, ρ = 1000–3500, c = 600–1200 imply Γ ≈ 2.4–2050 [read]; 2016 GE1 needs Γ ≲ 100, most probable ~20 [read]; bare rock ~2000+ (Delbo et al. 2015, arXiv:1508.05575 [verify]) |
| 11 | `ecc` | eccentricity at a = 1 au | 0–0.8 (q ≥ 0.2 au) | uniform | – | a = 1 au fixed by design; NEO orbits (Granvik et al. 2018, arXiv:1804.10265 [checked]); e > 0.6 flagged until CN stiffness at small q is validated |
| 12 | `D_eq_m` (STL scale) | volume-equivalent diameter | by stratum | log-uniform | m | strata below; spin barrier ~2.2 h for D ≳ 200 m [checked] |

Fixed: a = 1 au, Bond albedo 0.1, emissivity 0.9, c_p = 800 J kg⁻¹ K⁻¹ (irrelevant at fixed Γ, see C),
`version: new`, implicit Crank–Nicolson solver, convex ellipsoid meshes with 320 facets (icosphere, 2 subdivisions).

**Rotation strata (P_phi range, D range).** These are physically motivated and are not chosen for convenience:

| Stratum | P_phi | D | Literature anchor |
|---|---|---|---|
| slow | 24–1000 h | 20 m–10 km | Gaia excess of slow rotators (P ≳ 24 h), "mainly populated by tumblers" (Zhou et al. 2025, arXiv:2502.19631 [read p.1]); Toutatis LAM P_ψ ≈ 5.38 d, P_φ ≈ 7.40 d [checked]; Apophis SAM 27.374 h / 262.2 h (Ďurech et al. 2026 [read]); about 15 bodies with P > 1000 h (2019, secondary source [verify in LCDB]) |
| ordinary | 2.2–24 h | 20 m–10 km | spin barrier 2.2 h (Pravec & Harris 2000 [verify]) |
| fast | 10 min–2.2 h | 10–200 m | 156 of 161 TTT periods < 2.2 h (Alarcon et al. 2026 [read]); 2012 TC4 tumbler, main period 12.2 min, SAM (Lee et al. 2021, AJ 161, 112 [checked]) |
| super-fast | 5 s–10 min | 3–150 m | TTT shortest 5.44 s, H 22–29.8 → D ≈ 137–3.8 m at p_V = 0.15 [read]; 2008 TC3 tumbler 99.20 s / 97.00 s [checked]; 2016 GE1 34 s and 2011 PT 612 s (Marčeta et al. 2025 [read]) |

## B. How the eight rotational keys are sampled

| Key | Sampled variable | Constraint enforced | Role measured |
|---|---|---|---|
| `lambda_L`, `beta_L` | x₀, sin β_L | none needed | secular; λ_L only with e > 0 (e × λ_L interaction) |
| `I1`, `I2` | b/a, c/b (homogeneous ellipsoid) | one-to-one onto 0 < I1 ≤ I2 ≤ 1, I1 + I2 > 1 (proof in C) | secular (dynamics and surface) |
| `E_ratio` | balanced u ∈ (0, 1) | 1 < E < 1/I1; excluded bands: 0.1 % of the SAM range above 1, 1 % around the separatrix 1/I2, 0.1 % below 1/I1 | secular; SAM and LAM each get half of the design |
| `P_phi_h` | log P_phi in stratum | sets \|L\|; all other periods follow from (I1, I2, E) | secular |
| `phi_0` | uniform | none | finite-time and phase effects |
| `psi_0` | uniform on the reachable arcs | I₃A(ψ₀) ≥ E (otherwise the solver raises ValueError) | finite-time and phase effects |

**Recorded for every state** (derived with the solver's own `Simulation` and `mean_euler_rates`):
- mode (SAM or LAM) and θ₀ and θ range (the nutation);
- P_ψ, the polhode period T_ω, P_∥ = 2πL/2E, P_dt = 2π/max|ω| and the light-curve period P_lc;
- the ratios P_ψ/P_φ and T_ω/P_φ;
- the slowest low-order beat |p/T_ω − q/P_φ|⁻¹ (p, q ≤ 3) and its ratio to the orbital period.

**Averaging policy.** If any rotational timescale or beat exceeds 5 % of the orbital period, the state gets long
averaging: 30 orbits, with every orbit's drift stored. This measures interannual variation. Otherwise the run
uses converged averaging (4–12 orbits).

**PA twins.** Each tumbler gets a matched principal-axis twin: same shape, **L**, ρ, Γ, e and D, spinning about the
short axis with period P_∥. Study C adds twins at P_φ and P_lc (SAM only) for 256 states per stratum. Twins use the
tumbler's time step and physical depth grid. A twin's `P_phi_h` is P × I1 (the code's E = 1 convention), which the
tests confirm to 10⁻⁹.

## C. Density, c_p, K, Γ, diameter and shape

**Thermal grouping.**
- The code takes Γ, ρ and c_p, and derives K = Γ²/(ρc_p).
- In the 1D column with a depth grid in skin depths, temperatures depend on (ρ_s, c_p, K) only through Γ.
- [measured] Changing c_p from 600 to 1200 J kg⁻¹ K⁻¹ at fixed Γ and ρ changes the drift by 4.7 × 10⁻¹⁴ (relative).
- The proposed (ρ, C_pK) basis gives Γ = √(ρ·C_pK). Sampling ρ and C_pK independently would make ρ act twice, through the mass and through Γ, and would create an artificial ρ–Γ correlation.
- **Recommended:** sample bulk density and Γ independently. This is exactly the "separate dynamic and thermal influence" design in the group's own materials. C_pK = Γ²/ρ, K and κ = K/(ρc_p) are recorded for every state.
- K and κ matter separately only for finite bodies or layered regolith. The finite-sphere factor (Vokrouhlický 1999 linear theory, the repository's `Fnu_eval`) is stored per state, and states where it deviates by more than 5 % are flagged `outside_1d_validity`.

**Density and diameter.**
- [measured] Drift ∝ 1/ρ and ∝ 1/D exactly (errors 0 and 4 × 10⁻¹²; `research/runs/verification/raw/scaling`).
- Both are still sampled inputs, at no extra run cost. Their exact log-additive effect is a built-in check of the sensitivity analysis.
- D also controls validity (finite-body factor) and the spin-barrier constraint between strata.

**Parameter 12 = diameter.** Shape is already in the design through I1 and I2.
- For a homogeneous ellipsoid with semi-axes 1 ≥ b ≥ c: I1 = (b² + c²)/(1 + b²) and I2 = (1 + c²)/(1 + b²).
- Inverting gives b² = (1 − d)/(1 + d) and c² = 2I2/(1 + d) − 1, with d = I2 − I1.
- c² > 0 ⟺ I1 + I2 > 1, c ≤ b ⟺ I2 ≤ 1, and b ≤ 1 ⟺ I1 ≤ I2. So every admissible (I1, I2) is exactly one ellipsoid, and vice versa.
- Adding a/b or a/c as a 12th input would duplicate I1 and I2. The mesh inertia equals this formula to 10⁻¹⁰ (tests).
- Non-uniform interiors (inertia not following shape; for Apophis, mesh 0.73/0.95 vs dynamics 0.61/0.965) are a separate, later study.

**Independent dimensions.**
- 12 independent unit coordinates are sampled.
- In this 1D model, ρ and D enter the solver output only through exact 1/(ρD) scaling.
- φ₀ and ψ₀ (and λ_L when e = 0) are phases, expected to change only finite-time averages; this is tested, not assumed.
- That leaves 8 candidate secular dimensions (λ_L via e, β_L, I1, I2, E, P_φ, Γ, e), plus 2 scalings and 2 phases.

## D. Experimental design

| Study | Design | States | Runs (with twins) | Purpose |
|---|---|---|---|---|
| A | Morris (r = 20 trajectories, 8 levels, k = 12) per stratum | 260 + 260 | ~1,040 | screening, interaction flags (σ/μ*), noise and cost calibration |
| B | scrambled Sobol′ N = 1024 per stratum (extendable to 2048 with the same seed) | 1024 + 1024 | ~4,096 | surrogate (Gaussian process with nugget, or sparse polynomial chaos) and first/total Sobol indices with bootstrap intervals |
| C | first 256 Sobol′ points per stratum, twins at P_φ and P_lc | 512 | ~800 | effective-period conventions |
| D | 48 states × 8 random (φ₀, ψ₀) per stratum, all 30-orbit long averaging | 768 | ~864 | phase effects, finite-time and interannual variability, long- vs short-duration averages |
| E | Sobol′ N = 256 per stratum (fast, super-fast) | 512 | parameter sets only | waits for a validated multi-timescale solver (route flags in the manifest) |
| (F) | Saltelli N = 256 (3,584 points), slow stratum | 3,584 | optional | surrogate-free Sobol check, only if the surrogate fails validation |

**Sensitivity outputs.**
- Signed drift da/dt.
- log₁₀|da/dt| with the sign analysed separately as a classification boundary.
- The tumbling effect Δ = drift_tumbler − drift_twin, normalised by a scale that never vanishes: S = |diurnal(γ = 0)| + |seasonal(γ = 90°)| of the linear theory for the same D, ρ, Γ, P_∥. Plain ratios are not used near zero drift.
- The ratio to the twin, only where |drift_twin| > 10 S_num.
- Orbit-to-orbit standard deviation (study D).

**Numerical uncertainty.** Per run, the solver error estimate. Per state class, the study-D phase spread and the
convergence matrix, the latter scheduled as the first HPC job. Effects smaller than 3σ_num are reported as
unresolved. Morris uses μ* and σ. Sobol indices are computed on the surrogate only after a held-out Q² > 0.9, and the
sample is extended only while the interval half-widths of the leading total indices exceed 0.05.

## E. Solver feasibility (measured cost model, Apple M4, 1 thread)

t_step = 0.96 ms + 6.0 ns × facets × layers (tumbling; fit residuals < 8 %). The uniform grid needs
n_layers ≈ 45 √(P_orb/P_dt). The time step is reduced until the Crank–Nicolson surface stiffness S ≤ 1 (see
`research/CHANGELOG.md`, solver issue S-1).

| Stratum | Typical cost per run | Route with the current solver |
|---|---|---|
| slow | median 0.6 core-h, p90 8 | executable (`full`) |
| ordinary | median 2.7 core-h, p90 29 | executable; about 20 % `full_costly` (12–96 core-h) |
| fast (10 min–2.2 h) | median 41 core-h | 15 % `full`, about 50 % `full_costly`, 35 % multi-timescale required |
| super-fast (5 s–10 min) | 10³–10⁵ core-h | not executable: multi-timescale method required (frozen-Sun local steady state as in the IAC 2026 work, plus a seasonal coarse grid), validated against full integration in the overlap regime (P ≈ 1–6 h) |
