# Apophis after the 2029 Earth encounter: spin states and Yarkovsky drift

Inputs for Tempestovsky runs of the post-2029 Yarkovsky drift of (99942) Apophis, for all post-flyby spin
states of Benson et al. (2022, *Spin state evolution of (99942) Apophis during its 2029 Earth encounter*,
arXiv:2210.13365). Two independent reconstructions of the 50 000 post-flyby states are provided:

| set | how | keeps correlations between pole, periods and inertias |
|---|---|---|
| `sim` | the flyby is re-simulated with the setup of Sec. 3.1 of the paper | yes |
| `dig` | sampled from the digitized Figs. 2, 3 and 4 and the LAM fraction of the text | partly (not the pole) |

Row *k* of both input files has the same thermal and physical parameters, so the two sets differ only in the spin state.

## Flyby simulation (`flyby.py`, `ensemble.py`)

* Rigid body with principal moments I_l ≤ I_i ≤ I_s = 1, Earth gravity-gradient torque
  M = 3μ/R³ r̂ × (I r̂) along the JPL Horizons geocentric trajectory (`geo_vectors_2029.npy`,
  2029-04-11 to 04-17, 2 min steps, closest approach 38 011 km at JD 2462240.40709).
* Quaternion attitude, fixed-step RK4 (dt = 30 s, ±2 days around closest approach; dt = 60 s already agrees to 1e-9).
* Pre-flyby state as in the paper: nominal photometric periods P_φs = 27.38 h and P_ψ = 263 h,
  I_l/I_s uniform in [0.53, 0.73], I_i/I_s uniform in [0.95, 0.974] (pairs without a SAM state with these periods
  are rejected), pole uniform on the sphere inside the admissible region (red polygon of Fig. 4, `pole_region_pixels.npy`),
  uniform phase of ω in the body frame and uniform attitude phase about H.
* Torque-free rotation is handled analytically (Jacobi elliptic functions); the formulas agree with
  Tempestovsky's numerical `mean_euler_rates` to all printed digits.

### Validation against the paper

Fig. 7 (nominal pole and periods, 5000 runs for each inertia pair): the outer edge of the post-flyby pole cloud
lies within 0.1–0.5° (median) of the paper's outline for the same inertias, and 1–5° from the outlines of the
other inertias (`fig7_compare.png`).

Full ensemble (`compare_paper.py`, `figs_compare.png`):

| quantity | this work | Benson et al. |
|---|---|---|
| P_φl range / mean / most likely [h] | 16.8–70.1 / 31.0 / 20.9 | 16.4–67.2 / 30.9 / ~21 |
| P_ψ range / mean / most likely [h] | 97.6–1748.6 / 267.7 / 188 | 95.8–1686.9 / 266.0 / ~185 |
| P_φl decreases / P_ψ decreases / both | 55.6% / 58.9% / 51.0% | 55.8% / 59.3% / 51.2% |
| ω_e increases | 55.7% | 55.8% |
| LAM after the flyby | 16.6% | 17.3% |
| pole longitude increases / pole closer to the ecliptic | 91.6% / 55.4% | 93.5% / 55.3% |
| pole shift δ mean / most likely / max [deg] | 13.9 / 4.5 / 44.9 | 14.5 / ~5 / 45.8 |
| angular acceleration at closest approach, max / most likely [deg/h²] | 4.33 / 2.45 | 4.5 / 2.4 |
| cos γ_f / cos γ_i range, fraction < 1 | 0.64–1.23, 56% | 0.63–1.2, 56% |
| Eq. 6 drift ratio range, fraction < 1 | 0.59–1.12, 97% | 0.57–1.12, 97% |

The small remaining differences (slightly smaller pole shifts and accelerations) are partly due to the
closest-approach distance: the paper used the 2019 orbit solution (37 730 km), this work the current one (38 011 km),
and the torque scales as R⁻³. With the trajectory scaled to 37 730 km (`ca_distance_check.py`, 10 000 runs) the mean
δ goes from 13.92° to 14.27° (paper 14.5°).

`ensemble_summary.png`: post-flyby periods, pole shift against the digitized Fig. 5, and the obliquity and I_d
distributions of both sets. The figures that show the paper's own figures next to both sets, and the distributions
of all Tempestovsky inputs, are written by `make_local_figures.py` to `local_figures/` (ignored by git; the paper's
figures are not redistributed). The digitizers read the figure images from `figs/`
(`pdfimages -png -f 7 -l 11 <paper.pdf> figs/img`, also ignored by git).

## Digitized figures (`digitize_fig*.py`, `gen_digitized.py`)

* Fig. 2: the heatmap (120 × 240 bins, MATLAB `hot` colormap 0–100 runs) is decoded and raked to its two marginal
  histograms; the lowest colour level hides bins with 0–1 runs, the white diamond hides the pre-flyby bins.
* Fig. 4: pole density from the coverage of the scatter plot (Poisson overlap of ~30 px dots); the 30 saturated
  cells of the dense core are a lower limit. The projection (orthographic from the south ecliptic pole,
  r = 1327.2 px · cos β) reproduces the nominal and radar pole markers to 0.15°.
* Fig. 5 (δ histogram) is used as a check: digitized mean 14.48°, mode 4.5°, last bin 45.5°.
* Fig. 3: region and density (coverage) of the (ω_e, scaled I_d) scatter (`digitize_fig3.py`).
* Spin state: 150 000 candidates are built from Fig. 2 periods and uniform viable inertias (I_d from the period ratio,
  SAM and LAM branch where both exist). Their weights are raked so that together they reproduce the Fig. 2 periods,
  the Fig. 3 density (shape within SAM and within LAM) and the 17.3% LAM quoted in the text (residual mismatch 1.5%
  and 4.3%, effective sample size 88 000); 50 000 states are drawn by weight. Uniform inertias with an unweighted choice
  give 46–51% LAM: the inertias are uniform only a priori, LAM is much more likely for small I_i (Fig. 7).
* The pole is drawn independently from Fig. 4, so its correlation with the spin state is lost, and there is no
  pre-flyby pole per run (no δ, no Eq. 6 ratio).
* Compared with the re-simulated set: obliquity median 160.7° vs 161.0°, P_φs median 25.2 h vs 25.5 h,
  P_ψ median 232 h vs 235 h, E_ratio 97.5% point 1.084 vs 1.079, LAM 17.5% vs 16.6%.

## Tempestovsky inputs (`make_inputs.py` → `inputs_sim.csv`, `inputs_dig.csv`, committed gzipped)

| column | meaning |
|---|---|
| `I1`, `I2` | I_l/I_s, I_i/I_s |
| `E_ratio` | 2 I₃ E / L² = I_s / I_d |
| `P_phi_h` | mean precession period of the short axis about L (Tempestovsky's Euler angle convention) |
| `psi_0`, `phi_0` | Euler angles of the initial state (phases; do not matter for the orbit-averaged drift) |
| `lambda_L`, `beta_L` | post-flyby pole in the **post-flyby** orbital frame (x to perihelion, z orbit normal) |
| `a_au`, `ecc` | post-flyby orbit (JPL Horizons, 2029-05-30): a = 1.10312 au, e = 0.18907 |
| `thermal_inertia`, `albedo` | Monte Carlo as in `results/apophis_fig3` (Γ truncated to 100–1200) |
| `D`, `rho`, `roughness`, `shape_noise` | only rescale the drift afterwards (see `collect.py`) |
| `sam`, `P_psi_h`, `P_phi_l_h` | for reference (not read by Tempestovsky) |

The pole moves to the post-flyby orbital frame with the post-flyby elements: the argument of perihelion changes
from 126.7° to 71.4°, so the same pole has a λ_L about 55° larger than before the flyby.

## Orbit change alone

Same spin state and the same pole in J2000 ecliptic (250°, −75°; nominal photometric state, Γ = 600, A = 0.14),
on the orbit before and after the flyby: −8.0916e-6 m/s → −7.4969e-6 m/s, ratio 0.927. The diurnal linear theory
(Eq. 6 of the paper) gives √(a_i/a_f) · cos γ_f / cos γ_i = 0.914 · 0.997 = 0.912 for this case.

## Running the batch (`run_batch.py`, `collect.py`)

Setting up a server (Python 3.10, as used here):

```bash
git clone -b tumbling https://github.com/dusanmarceta/Jarkovsky-Tumbling.git
cd Jarkovsky-Tumbling
python3 -m venv venv && source venv/bin/activate      # or a conda environment
pip install numpy scipy numba astropy matplotlib pandas joblib numpy-stl tqdm pyyaml h5py
# one run first: checks the installation and fills the numba cache before many processes start at once
python results/apophis_flyby_2029/run_batch.py results/apophis_flyby_2029/inputs_sim.csv.gz --out runs_sim --last 1
cat runs_sim/run_00000.txt                            # drift, valid: True, converged: True
```


From the repository root, one single-threaded Tempestovsky process per run, round-robin chunks:

```bash
# one chunk per processor or per job of a job array (chunk = 0..99)
python results/apophis_flyby_2029/run_batch.py results/apophis_flyby_2029/inputs_sim.csv.gz \
    --out runs_sim --chunk $CHUNK --n-chunks 100
# or everything on one machine with W parallel workers
python results/apophis_flyby_2029/run_batch.py results/apophis_flyby_2029/inputs_sim.csv.gz --out runs_sim --workers 6
# afterwards
python results/apophis_flyby_2029/collect.py results/apophis_flyby_2029/inputs_sim.csv.gz runs_sim \
    results/apophis_flyby_2029/inputs_dig.csv.gz runs_dig
```

Existing outputs are skipped, so an interrupted batch can be restarted. Extra config overrides:
`--set implicit_timesteps_per_day=100`.

### Settings and cost

The defaults of `run_batch.py` are `implicit_timesteps_per_day: 100`, `yarkovsky_tolerance: 0.005`, `max_orbits: 10`:

* 100 instead of 200 steps per rotation changes the drift by 2e-5 to 9e-4 (relative) in the tested post-flyby states
  (fast SAM, deep LAM, typical, and the same at Γ = 100).
* For a tumbling body the drift of single orbits scatters by 0.2–0.4% because every orbit sees a different sequence of
  rotation phases (the orbital period is not commensurate with P_φ and P_ψ). With a 1e-3 tolerance the run stops only
  when two consecutive orbits agree by chance, which took 12–20 orbits for a deep LAM state at Γ = 100. With 0.5% the
  runs stop after the thermal transient (4 orbits in all tests); the scatter of 0.2–0.4% is far below the spread of
  the ensemble.

Time step and skin depth are set by 2π/max|ω| (16–56 h for the post-flyby states, median 28 h). With 100 steps per
rotation an orbit (423 d) has 37 000 steps on average (up to 104 000). Measured on an i5-9500 with all 6 cores busy,
one run takes 290–360 s; 50 000 runs are about 4 000 core-hours, i.e. **~40 h on 100 cores per set**. The rows are in
random order, so the first 10 000 rows (`--last 10000`, ~8 h on 100 cores) already give the distribution
(standard error of the mean drift ~0.3%). Memory: ~0.6 GB per process.
