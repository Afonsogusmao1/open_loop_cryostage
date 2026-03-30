Open_loop workspace

This folder contains the data, calibrated baseline simulations, and active code for an implemented offline open-loop workflow for controlling freezing-front evolution in the cryostage system.

The repository now serves two connected purposes:
1. keep the experimental and calibrated reference material used to validate the models,
2. run and analyze open-loop trajectory studies of the cascade
   theta -> T_ref(t) -> cryostage model -> T_plate(t) -> solver -> z_front(t) -> J(theta)

The exploratory 180/240/360 s phase should be treated as completed scoping work.
The next article-facing phase is a single full-process optimization carried through freeze completion / total solidification using the solver's existing freeze-complete logic.

==================================================
FOLDER ORGANIZATION
==================================================

Open_loop
|
|-- data
|   |
|   |-- characterization_cryostage
|   |   |
|   |   |-- characterization_min5
|   |   |-- characterization_min10
|   |   |-- characterization_min15
|   |   |-- characterization_min20
|   |   |-- cryostage_characterization
|   |   |-- figures
|   |   |-- plot_characterization_assays.py
|   |
|   |-- constant_plateT_water_ICT_readings
|   |   |
|   |   |-- min10
|   |   |-- min15
|   |   |-- min20
|   |   |-- overlay_probe_runs.py
|   |   |-- plot_run_csv.py
|   |
|   |-- simulations_calibrated
|       |
|       |-- figures
|       |-- simulation case folders
|
|-- code_simulation
|   |
|   |-- cryostage_model.py
|   |-- front_tracking.py
|   |-- geometry.py
|   |-- materials.py
|   |-- open_loop_cascade.py
|   |-- open_loop_optimizer.py
|   |-- open_loop_problem.py
|   |-- run_calibration_fixed_h.py
|   |-- run_open_loop_optimization.py
|   |-- run_open_loop_study.py
|   |-- run_optimizer_learning_diagnostics.py
|   |-- solver.py
|   |-- trajectory_profiles.py
|   |-- results
|       |-- cryostage_model_validation
|       |-- open_loop_optimization
|       |-- open_loop_study

==================================================
DATA FOLDER
==================================================

The data folder stores the experimental datasets, processed results, baseline calibrated simulations, and plotting utilities used in the current workflow.

--------------------------------------------------
1. characterization_cryostage
--------------------------------------------------

This folder contains the cryostage characterization datasets.

Its purpose is to describe the thermal response of the cryostage plate when a temperature setpoint is imposed and tracked by the calibrated PID controller.

The folders:
- characterization_min5
- characterization_min10
- characterization_min15
- characterization_min20

contain the CSV files for the repeated cryostage runs performed at setpoints of -5 C, -10 C, -15 C, and -20 C, respectively.

These runs are used to study how the measured plate temperature evolves in time after a commanded setpoint change.

Additional contents:
- cryostage_characterization
  Contains firmware and related material for the cryostage characterization setup.
- figures
  Contains saved figures for the characterized temperature responses.
- plot_characterization_assays.py
  Script used to process and visualize the characterization assays.

This dataset is intended to support the identification of a reduced cryostage response model that maps:

T_ref(t) -> T_plate(t)

where:
- T_ref(t) is the commanded cryostage reference temperature,
- T_plate(t) is the measured plate temperature.

This data is already being used by the current reduced cryostage model, and saved validation artifacts are present under code_simulation/results/cryostage_model_validation/.

--------------------------------------------------
2. constant_plateT_water_ICT_readings
--------------------------------------------------

This folder contains the experimental water freezing runs performed on the cryostage with approximately constant base temperature conditions.

Its purpose is to store the thermocouple measurements collected during directional freezing experiments, which are used for calibration and validation of the 2D thermal model.

The folders:
- min10
- min15
- min20

contain the experimental runs performed with the cryostage set approximately to -10 C, -15 C, and -20 C.

Each of these folders contains raw CSV files from the thermocouples, as well as a results subfolder with processed outputs such as plots and processed CSV files.

This directory also contains:
- overlay_probe_runs.py
- plot_run_csv.py

which are scripts used to process and visualize the thermocouple data.

There is also presentation material containing plots of the experimental readings and model-versus-experiment comparisons.

These data are used to compare the internal temperature evolution measured experimentally with the temperatures predicted by the calibrated 2D simulation.

--------------------------------------------------
3. simulations_calibrated
--------------------------------------------------

This folder contains the outputs of the calibrated 2D simulations.

Its purpose is to keep the baseline numerical results obtained after calibration against the experimental freezing data.

The simulation case folders contain outputs such as:
- probe temperature CSV files,
- front tracking CSV files,
- curved front CSV files,
- and ParaView-compatible simulation files such as XDMF and H5.

This folder also contains a figures folder with:
- comparison plots of simulated versus experimental runs,
- and the scripts used to generate those plots.

These results represent the current calibrated baseline of the simulation framework and should be treated as the reference state before introducing open-loop trajectory design.
They remain the baseline reference state for the current open-loop work as well.

==================================================
CODE_SIMULATION FOLDER
==================================================

The code_simulation folder is the active development area of the project.

This is where the implemented offline open-loop workflow now lives and where the next full-freezing phase should be developed.

Core simulation and optimization modules include:
- solver.py
  Main 2D thermal / phase-change simulation code.
- geometry.py
  Geometry definitions and boundary-condition related setup.
- materials.py
  Material property definitions and thermophysical parameters.
- front_tracking.py
  Utilities for extracting and analysing freezing-front evolution.
- cryostage_model.py
  Reduced cryostage dynamics used to map T_ref(t) to T_plate(t).
- open_loop_cascade.py
  Glue code for the full open-loop cascade from T_ref(t) to solver outputs.
- open_loop_problem.py
  Front-position-based objective definition and reference construction.
- open_loop_optimizer.py
  Optimizer wrapper for the open-loop parameter search.

Active scripts already present include:
- run_calibration_fixed_h.py
  Baseline fixed-temperature calibration / reference simulation runner.
- run_open_loop_optimization.py
  Compact reproducible open-loop optimization runner.
- run_open_loop_study.py
  Study harness that writes summaries, histories, best-run artifacts, and standard plots.
- run_optimizer_learning_diagnostics.py
  Post-processing script for optimizer-learning and cross-study comparison figures.

The folder also contains:
- results
  Stores saved cryostage-model validation artifacts, open-loop optimization runs, and open-loop study outputs.

==================================================
CURRENT WORKFLOW
==================================================

The current active workflow is:

theta -> T_ref(t) -> cryostage model -> T_plate(t) -> 2D freezing solver -> z_front(t) -> J(theta)

The objective remains based on front-position behaviour and front-reference tracking.
It is not based on direct raw front-velocity control.

The current workspace therefore combines four connected levels:
1. Cryostage characterization data and reduced-model validation.
2. Experimental water-freezing measurements.
3. Calibrated baseline 2D simulations.
4. Implemented open-loop optimization and study tooling.

==================================================
EXPLORATORY RESULTS STATUS
==================================================

The short-horizon exploratory phase is already in hand and should be treated as completed scoping work.

The current snapshot includes:
- saved 180 s validation outputs under code_simulation/results/open_loop_study/validation_h180_k5_nm/
- multiple 360 s feasibility, sensitivity, optimizer-learning, and article-package outputs under code_simulation/results/open_loop_study/
- a 600 s feasibility extension under code_simulation/results/open_loop_study/vanilla_feasibility_h600_k5_nm_ti0/
- cryostage-model validation artifacts dated 2026-03-29 under code_simulation/results/cryostage_model_validation/

The reusable study runner still exposes a 240 s study configuration, but a dedicated saved h240 study folder is not present in this snapshot.

==================================================
NEXT DEVELOPMENT STAGE
==================================================

The next objective is not to build the open-loop workflow from scratch.
The next objective is to move the existing workflow from exploratory short-window studies to a full-freezing / total-solidification formulation.

The intended cascade is:

theta -> T_ref(t) -> cryostage model -> T_plate(t) -> 2D freezing model -> z_front(t) -> J(theta)

where:
- theta represents the trajectory parameters,
- T_ref(t) is the cryostage reference temperature trajectory,
- T_plate(t) is the actual plate temperature predicted from cryostage dynamics,
- z_front(t) is the freezing-front position,
- J(theta) is an objective function based on front-position behaviour and tracking, not raw front velocity.

The solver already contains freeze-complete handling for the filled region through FreezeStopOptions(mode="fillable_region") and the recorded freeze_complete_flag.
The next phase should reuse that existing logic.

The intended formulation is one full-process open-loop optimization that runs from fill until solver-detected freeze completion.
The goal is not real-time feedback control of the freezing front, and it is not another round of short-window exploratory studies.

==================================================
GENERAL NOTE
==================================================

The data folder should be regarded as the repository of experimental reference data, processed measurements, and baseline simulation outputs.

The code_simulation folder should be regarded as the main area for active model development, saved study outputs, and the next full-process open-loop phase.