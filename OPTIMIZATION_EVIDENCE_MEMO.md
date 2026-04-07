# Optimization Evidence Memo

## Workflow objective
The workflow objective is model-based open-loop design of `T_ref(t)` for the inner PID to obtain approximately linear freezing-front progression.

## Evidence source studies
This memo uses existing outputs only from the following completed optimization studies.

- Schedule sensitivity study:
  - `code_simulation/results/open_loop_bayesian_optimization/schedule_sensitivity_first_pass_n3_n4/`
- Uniform multiseed confirmation study:
  - `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed17/`
  - `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed29/`
  - `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed41/`

## Schedule sensitivity study
The schedule sensitivity study compared `N=3` and `N=4` under the same BO and admissibility policy while changing only the externally fixed knot-time schedule.

- `uniform`:
  - `N=3`: `2.538915394e-01`
  - `N=4`: `2.533405138e-01`
  - Winner: `N=4`
- `early_dense`:
  - `N=3`: `2.588256647e-01`
  - `N=4`: `3.035365257e-01`
  - Winner: `N=3`
- `late_dense`:
  - `N=3`: `2.711955936e-01`
  - `N=4`: `2.566860378e-01`
  - Winner: `N=4`

Conclusion: the `N=3` versus `N=4` ranking is schedule-dependent under the current workflow.

## Uniform multiseed confirmation
The uniform multiseed confirmation study compared `N=3` and `N=4` under the default external schedule `uniform`, with shared settings `init_points=6`, `n_iter=16`, and BO seeds `17`, `29`, and `41`.

### Seed-by-seed results
| Seed | `N=3` best objective | `N=4` best objective | Winner |
| --- | ---: | ---: | --- |
| `17` | `2.545899717e-01` | `2.473153266e-01` | `N=4` |
| `29` | `2.538915394e-01` | `2.729151682e-01` | `N=3` |
| `41` | `2.538915394e-01` | `2.464709220e-01` | `N=4` |

### Feasibility and early rejection counts
| Seed | `N` | Feasible evaluations | Early rejections |
| --- | ---: | ---: | ---: |
| `17` | `3` | `22` | `1` |
| `17` | `4` | `20` | `3` |
| `29` | `3` | `22` | `1` |
| `29` | `4` | `21` | `2` |
| `41` | `3` | `22` | `1` |
| `41` | `4` | `20` | `3` |

### Aggregate comparison
- `N=3` mean best objective: `2.541243501e-01`
- `N=3` median best objective: `2.538915394e-01`
- `N=4` mean best objective: `2.555671389e-01`
- `N=4` median best objective: `2.473153266e-01`

Conclusion: `N=4` shows upside under `uniform`, but it does not yet show robust superiority. It wins in two seeds, loses clearly in one seed, and is less stable than `N=3`.

## Current workflow status
- Default external knot-time schedule: `uniform`
- Pragmatic working default for ongoing engineering use: `uniform + N=3`
- Unresolved challenger for ongoing workflow development: `uniform + N=4`
- Scientific conclusion status: knot count remains unresolved under the current workflow

## What this does and does not imply
- This evidence does support using the current BO-based workflow for ongoing development, with `uniform` as the default external schedule and `uniform + N=3` as the pragmatic working default.
- This evidence does not support claiming the globally best knot count.
- This evidence does not support claiming the globally best external time discretization.
- These results should be read as sensitivity analysis of externally fixed workflow design choices under the current admissibility and BO setup.

## Evidence figures
A small evidence-figure folder was copied from existing study outputs only:

- [optimization_evidence_figures](/home/fenics/shared/Open_loop/optimization_evidence_figures)

The copied figures preserve the original study outputs and are named to indicate their study source and winner context. No new figures were rendered for this package.
