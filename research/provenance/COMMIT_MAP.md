# Commit map: original local history → GitHub-based history

The research work was first committed in a local repository whose root commit `f113ce6` was a copy of
`tumbling` at `3fef255` without Git metadata. Branch `research/superfast-method-github` replays the same
commits on top of GitHub's `tumbling` at `3fef255`. Every replayed commit has the same message and the same
files as its original. The only difference: the copy did not track six files excluded by `.gitignore`
(`documentation/readme/geometry.svg` and five `*.npz` under `results/`), which are kept here as on GitHub.

Hashes cited in `research/CHANGELOG.md`, `research/FREEZE.md` and `research/proposals/` refer to the original
history (left column).

| Original | GitHub-based | Commit |
|---|---|---|
| `f113ce6` | `b10b471` | baseline: repository as received (checksums in `research/provenance/baseline_sha256.txt`) |
| `0fe8c2b` | `0cdd951` | research: environment record, run wrapper, baseline reproductions, generator modules |
| `abfbae9` | `322f35e` | research/tests: rotation, period-convention, shape-inertia and generator verification (31 tests) |
| `2af6818` | `d10b49e` | research: 12-input campaign generator, HPC layer, solver issue S-1 report |
| `0302757` | `3a7afef` | research: fast rotation-state derivation, resumable build, campaign manifests A-E and V |
| `64fcaa1` | `d5a14d6` | research: stop tracking raw .jsonl manifests (archived as .jsonl.gz; probe/smoke files are scratch) |
| `e6a6390` | `69c250d` | research: record built manifest counts and budget in PARAMETER_SPACE.md |
| `fc9b41d` | `6856074` | solver: S-1 fix - safeguarded surface Newton solve and physical temperature bound |
| `f5b3e0a` | `ee3b236` | research: S-1 validation suite, evidence and change record |
| `e9453bb` | `ade4ffb` | research: freeze record for HPC deployment (hpc-freeze-v1) |
| `cecb8ae` | `193ad9f` | research: execution waves, outside-1D exclusion list and V2 convergence manifest |
| `5d5fcfc` | `439958b` | superfast: two-scale (frozen-Sun diurnal + seasonal) method, work in progress |
| `ab848a3` | `540aea1` | superfast: validated two-scale method for the deferred fast and super-fast runs |
| `1798c84` | `32da083` | superfast: convergence study of the near-resonant 55.6 s super-fast tumbler |

Tag `hpc-freeze-v1` (original history) points to `e9453bb`. Its equivalent here is the commit in the
`e9453bb` row; the tag itself has not been recreated.
