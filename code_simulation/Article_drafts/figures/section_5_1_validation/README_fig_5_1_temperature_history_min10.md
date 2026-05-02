# fig_5_1_temperature_history_min10

## Files

- `fig_5_1_temperature_history_min10.png`
- `fig_5_1_temperature_history_min10.pdf`

## Data Sources

Experimental CSV files:

- `data/constant_plateT_water_ICT_readings/min10/cryostage_log_min10_20260303_112829.csv`
- `data/constant_plateT_water_ICT_readings/min10/cryostage_log_min10_20260303_123511.csv`
- `data/constant_plateT_water_ICT_readings/min10/cryostage_log_min10_20260303_135735.csv`
- `data/constant_plateT_water_ICT_readings/min10/cryostage_log_min10_20260303_145342.csv`
- `data/constant_plateT_water_ICT_readings/min10/cryostage_log_min10_20260306_140738.csv`

Simulation CSV file:

- `data/simulations_calibrated/with_rho_temperature_dependent/plate_m10_probe_stabilized_Tamb_9p7078_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm/water_PLA_calib_plate_m10_probe_stabilized_Tamb_9p7078_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm_probes.csv`

Metadata:

- `data/simulations_calibrated/with_rho_temperature_dependent/plate_m10_probe_stabilized_Tamb_9p7078_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm/water_PLA_calib_plate_m10_probe_stabilized_Tamb_9p7078_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm_metadata.csv`

## Method

- Experimental runs were aligned to the fill/insertion event using the same derivative-based onset detection as the calibrated comparison plotting workflow.
- The five aligned experimental runs were interpolated onto a common time base.
- Colored lines show the experimental mean and shaded bands show +/- 1 SD.
- The simulation trace comes from `data/simulations_calibrated/with_rho_temperature_dependent`.
- Time is plotted with the fill/insertion event displayed at 20 s to preserve the pre-fill context.

## Caption

Experimental and simulated temperature histories for the constant-plate -10 deg C validation experiment. Thin gray lines are the five experimental water-fill runs, colored lines and shaded bands show the experimental mean +/- 1 SD at each probe height, and the dashed black line is the calibrated rho(T) simulation using Tfill = 12.5 deg C and h = 2.0 W m^-2 K^-1.

## Interpretation Notes

- This figure is a validation check for the constant-temperature baseline, not an optimized trajectory result.
- The model captures the dominant cooling transient and the ordering of the three probe responses.
- Remaining late-stage mismatch, especially at upper probe heights, should be discussed as a limitation of the conduction-dominated model and not as a failure of the open-loop optimization framework.
