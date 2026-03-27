Open_loop workspace

This folder contains the data and code used to develop an offline open-loop framework for controlling freezing-front evolution in the cryostage system.

The project combines:
1. cryostage characterization data,
2. experimental thermocouple readings from water freezing tests,
3. calibrated 2D simulation outputs,
4. and the active simulation code used for further development.

==================================================
FOLDER ORGANIZATION
==================================================

Open_loop
|
|-- data
|   |
|   |-- characterization_cryostage
|   |   |
|   |   |-- characterizations_min5
|   |   |-- characterizations_min10
|   |   |-- characterizations_min15
|   |   |-- characterizations_min20
|   |   |-- cryostage_characterization
|   |   |-- figures
|   |
|   |-- constant_plateT_water_ICT_readings
|   |   |
|   |   |-- min10
|   |   |-- min15
|   |   |-- min20
|   |   |-- overlay_probe_runs
|   |   |-- plot_run_csv
|   |
|   |-- simulations_calibrated
|       |
|       |-- figures
|       |-- simulation case folders
|
|-- code_simulation
|   |
|   |-- front_tracking.py
|   |-- geometry.py
|   |-- materials.py
|   |-- solver.py
|   |-- other code files
|   |-- results

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
- characterizations_min5
- characterizations_min10
- characterizations_min15
- characterizations_min20

contain the CSV files for the repeated cryostage runs performed at setpoints of -5 C, -10 C, -15 C, and -20 C, respectively.

These runs are used to study how the measured plate temperature evolves in time after a commanded setpoint change.

Additional contents:
- cryostage_characterization
  Contains firmware and related material for the cryostage characterization setup.
- figures
  Contains scripts and figures used to visualize the characterized temperature responses for all target temperatures and replicates.

This dataset is intended to support the identification of a reduced cryostage response model that maps:

T_ref(t) -> T_plate(t)

where:
- T_ref(t) is the commanded cryostage reference temperature,
- T_plate(t) is the measured plate temperature.

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
- overlay_probe_runs
- plot_run_csv

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

==================================================
CODE_SIMULATION FOLDER
==================================================

The code_simulation folder is the active development area of the project.

This is where the simulation code is further developed and where future implementation of the offline open-loop framework will take place.

Typical files include:
- solver.py
  Main 2D thermal / phase-change simulation code.
- geometry.py
  Geometry definitions and boundary-condition related setup.
- materials.py
  Material property definitions and thermophysical parameters.
- front_tracking.py
  Utilities for extracting and analysing freezing-front evolution.

The folder also contains:
- results
  Intended to store newly generated outputs from ongoing code development, testing, and future optimization workflows.

==================================================
CURRENT WORKFLOW
==================================================

At the current stage, the project is organized into three connected levels:

1. Cryostage characterization
   The dynamic thermal response of the cryostage is measured under PID-controlled setpoint changes.

2. Experimental freezing data
   Water freezing experiments are performed and thermocouple temperatures are recorded at different heights in the mold.

3. Calibrated 2D simulation
   A 2D numerical model is used to reproduce the freezing experiments and generate simulation outputs for comparison with the measured data.

==================================================
NEXT DEVELOPMENT STAGE
==================================================

The next objective is to extend this workspace toward offline open-loop trajectory design.

The intended cascade is:

theta -> T_ref(t) -> cryostage model -> T_plate(t) -> 2D freezing model -> z_front(t) -> J(theta)

where:
- theta represents the trajectory parameters,
- T_ref(t) is the cryostage reference temperature trajectory,
- T_plate(t) is the actual plate temperature predicted from cryostage dynamics,
- z_front(t) is the freezing-front position,
- J(theta) is an objective function based on front-position tracking.

The goal is not real-time feedback control of the freezing front.
The goal is offline design of physically plausible cryostage temperature trajectories capable of producing a desired freezing-front evolution.

==================================================
GENERAL NOTE
==================================================

The data folder should be regarded as the repository of experimental reference data, processed measurements, and baseline simulation outputs.

The code_simulation folder should be regarded as the main area for active model development and future implementation of the open-loop framework.