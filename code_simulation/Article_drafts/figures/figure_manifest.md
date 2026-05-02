# Article figure manifest

Generated on: 2026-04-30

## Sources

- BO campaign: `code_simulation/results/active/bo_velocity_control/n8/diagnostico/earlydense_directspeed_w50_tol1_robust_0p007_0p013_5seed`
- Campaign summary: `code_simulation/results/active/bo_velocity_control/n8/diagnostico/earlydense_directspeed_w50_tol1_robust_0p007_0p013_5seed/campaign_summary.csv`
- Objective evaluations: `code_simulation/results/active/bo_velocity_control/n8/diagnostico/earlydense_directspeed_w50_tol1_robust_0p007_0p013_5seed/objective_by_evaluation.csv`
- Fine summary: `code_simulation/results/active/bo_velocity_control/n8/diagnostico/earlydense_directspeed_w50_tol1_robust_0p007_0p013_5seed/fine_confirmations_best_per_target_full_process_article/fine_confirmation_summary.csv`

## Counts

- Completed coarse BO rows: `35`
- Coarse targets: `7`
- Coarse seeds: `5`
- Completed fine representative rows: `7`
- Fine representative targets: `7`
- Example targets for time-series figures: `0.007, 0.010, 0.013 mm/s`

## Figures and captions

### `fig_bo_target_vs_achieved_n8`

- File: `code_simulation/Article_drafts/figures/fig_bo_target_vs_achieved_n8.png`
- File: `code_simulation/Article_drafts/figures/fig_bo_target_vs_achieved_n8.pdf`
- Caption: Coarse n8 robust Bayesian optimization results: target direct speed against achieved direct speed for all five seeds. The dashed line marks perfect agreement.

### `fig_bo_objective_phases_n8`

- File: `code_simulation/Article_drafts/figures/fig_bo_objective_phases_n8.png`
- File: `code_simulation/Article_drafts/figures/fig_bo_objective_phases_n8.pdf`
- Caption: Objective-function convergence during the n8 robust BO campaign. Raw objective evaluations are shown as transparent points, best-so-far curves as solid lines, and shaded bands mark theta0, deterministic initialization, BO acquisition, and local refinement.

### `fig_bo_error_vs_target_seed_variability`

- File: `code_simulation/Article_drafts/figures/fig_bo_error_vs_target_seed_variability.png`
- File: `code_simulation/Article_drafts/figures/fig_bo_error_vs_target_seed_variability.pdf`
- Caption: Direct-speed error and across-seed spread after robust n8 BO. This is the main diagnostic for seed variability in the coarse optimization stage.

### `fig_fine_target_vs_achieved_best_per_target`

- File: `code_simulation/Article_drafts/figures/fig_fine_target_vs_achieved_best_per_target.png`
- File: `code_simulation/Article_drafts/figures/fig_fine_target_vs_achieved_best_per_target.pdf`
- Caption: Fine-grid confirmation of the selected best seed for each target. These data validate whether the direct-speed result survives the full-process article simulation profile.

### `fig_temperature_tracking_examples`

- File: `code_simulation/Article_drafts/figures/fig_temperature_tracking_examples.png`
- File: `code_simulation/Article_drafts/figures/fig_temperature_tracking_examples.pdf`
- Caption: Representative examples of reference and achieved plate temperature during fine confirmations. Shaded regions indicate the front-tracking evaluation window.

### `fig_front_position_tracking_examples`

- File: `code_simulation/Article_drafts/figures/fig_front_position_tracking_examples.png`
- File: `code_simulation/Article_drafts/figures/fig_front_position_tracking_examples.pdf`
- Caption: Representative fine-grid freezing-front position trajectories compared with the linear target reference over the controlled depth interval.

### `fig_front_velocity_tracking_examples`

- File: `code_simulation/Article_drafts/figures/fig_front_velocity_tracking_examples.png`
- File: `code_simulation/Article_drafts/figures/fig_front_velocity_tracking_examples.pdf`
- Caption: Representative fine-grid freezing-front velocity trajectories. The simulated velocity is shown as a 30 s moving mean to expose the control trend rather than numerical differentiation noise.

## Notes for Oliveira

- The objective plot directly addresses whether BO was still improving when each phase ended.
- The seed-variability figure separates direct-speed error from across-seed achieved-speed spread.
- The fine figures are representative best-per-target confirmations, not yet the full all-seed fine robustness set.
- The Word draft was not edited in this pass.
