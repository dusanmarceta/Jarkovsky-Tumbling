# Proposed solver correction S-1 (not applied)

File: `src/model/solvers/tempestovsky_implicit.py`, function `surface_temperature`, nonlinear branch.

## Current code

```python
x = T0
for _ in range(50):
    f = lin * x + nl * x**4 - R
    df = lin + 4.0 * nl * x**3
    dx = f / df
    x -= dx
    if abs(dx) < 1e-10:
        break
return x
```

## Proposed code

```python
if R <= 0.0:                 # lin*x + nl*x^4 = R has no positive root: tell the caller to use backward Euler
    return np.nan
# both R/lin and (R/nl)^(1/4) bound the root from above; f is increasing and convex for x > 0, so Newton from
# the smaller bound decreases monotonically to the root (no overshoot, no wandering)
x = min(R / lin, (R / nl) ** 0.25)
for _ in range(100):
    f = lin * x + nl * x**4 - R
    dx = f / (lin + 4.0 * nl * x**3)
    x -= dx
    if abs(dx) <= 1e-12 * x:
        return x
return np.nan                # not converged: caller falls back to backward Euler
```

`implicit_column_step` already treats a NaN result like a negative one (`if not x >= 0.0`). The backward-Euler
equation always has R > 0 (no explicit emission term), so the same safeguarded solve has a root there.

Also proposed, in `tempestovsky.py`, `YarkovskySolver.solve`: set `valid: False` when the surface temperature
exceeds 1.05 × the perihelion subsolar temperature, and report the fraction of fallback steps.

## Expected effect and checks before merging

- Whenever the old Newton converged, the root is the same, so results agree to round-off. Re-run the two baseline
  reproductions (flyby run 0, linear_comparison) and require agreement to 1e-10.
- Stiff cases: the benchmark case `pa_f1280_L1200` must become finite, with T_max below the bound.
- Cost: none measurable (the iteration count drops because the start is monotone).
