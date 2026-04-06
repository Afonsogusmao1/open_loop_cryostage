# Current Workflow Status

The current recommended open-loop workflow is the `N = 3` Bayesian-optimization path validated by the focused confirmation rerun.

- Recommended single-run optimizer entry point: `run_open_loop_optimization.py`
- Recommended fixed-`N` comparison study entry point: `run_open_loop_fixed_n_bo_study.py`
- Recommended admissibility diagnostics entry point: `run_reachability_diagnostics.py`
- Recommended final export/locking entry point: `export_final_locked_n3_workflow.py`

The locked final artifact bundle for the selected workflow is written under:

- `code_simulation/results/open_loop_final_workflow/locked_n3_seed29_init4_iter8/`

That folder contains the selected `theta`, the exported reference trajectory, the front-tracking and plate/reference plots, the final admissibility summary, and a concise workflow guide.
