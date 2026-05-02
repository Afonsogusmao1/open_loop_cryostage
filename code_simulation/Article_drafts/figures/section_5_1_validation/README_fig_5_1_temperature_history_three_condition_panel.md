# fig_5_1_temperature_history_three_condition_panel

## Files

- `fig_5_1_temperature_history_three_condition_panel.png`
- `fig_5_1_temperature_history_three_condition_panel.pdf`

## Data Sources

Experimental CSV files:

- `data/constant_plateT_water_ICT_readings/min10/cryostage_log_min10_20260303_112829.csv`
- `data/constant_plateT_water_ICT_readings/min10/cryostage_log_min10_20260303_123511.csv`
- `data/constant_plateT_water_ICT_readings/min10/cryostage_log_min10_20260303_135735.csv`
- `data/constant_plateT_water_ICT_readings/min10/cryostage_log_min10_20260303_145342.csv`
- `data/constant_plateT_water_ICT_readings/min10/cryostage_log_min10_20260306_140738.csv`
- `data/constant_plateT_water_ICT_readings/min15/cryostage_log_min15_20260305_141135.csv`
- `data/constant_plateT_water_ICT_readings/min15/cryostage_log_min15_20260305_151423.csv`
- `data/constant_plateT_water_ICT_readings/min15/cryostage_log_min15_20260305_161931.csv`
- `data/constant_plateT_water_ICT_readings/min15/cryostage_log_min15_20260305_174008.csv`
- `data/constant_plateT_water_ICT_readings/min15/cryostage_log_min15_20260305_185049.csv`
- `data/constant_plateT_water_ICT_readings/min20/cryostage_log_min20_20260304_153956.csv`
- `data/constant_plateT_water_ICT_readings/min20/cryostage_log_min20_20260304_163300.csv`
- `data/constant_plateT_water_ICT_readings/min20/cryostage_log_min20_20260304_174007.csv`
- `data/constant_plateT_water_ICT_readings/min20/cryostage_log_min20_20260304_195718.csv`
- `data/constant_plateT_water_ICT_readings/min20/cryostage_log_min20_20260305_130922.csv`

Simulation CSV files:

- `data/simulations_calibrated/with_rho_temperature_dependent/plate_m10_probe_stabilized_Tamb_9p7078_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm/water_PLA_calib_plate_m10_probe_stabilized_Tamb_9p7078_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm_probes.csv`
- `data/simulations_calibrated/with_rho_temperature_dependent/plate_m15_probe_stabilized_Tamb_7p568_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm/water_PLA_calib_plate_m15_probe_stabilized_Tamb_7p568_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm_probes.csv`
- `data/simulations_calibrated/with_rho_temperature_dependent/plate_m20_probe_stabilized_Tamb_5p75_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm/water_PLA_calib_plate_m20_probe_stabilized_Tamb_5p75_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm_probes.csv`

Metadata:

- `data/simulations_calibrated/with_rho_temperature_dependent/plate_m10_probe_stabilized_Tamb_9p7078_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm/water_PLA_calib_plate_m10_probe_stabilized_Tamb_9p7078_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm_metadata.csv`
- `data/simulations_calibrated/with_rho_temperature_dependent/plate_m15_probe_stabilized_Tamb_7p568_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm/water_PLA_calib_plate_m15_probe_stabilized_Tamb_7p568_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm_metadata.csv`
- `data/simulations_calibrated/with_rho_temperature_dependent/plate_m20_probe_stabilized_Tamb_5p75_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm/water_PLA_calib_plate_m20_probe_stabilized_Tamb_5p75_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm_metadata.csv`

## Method

- This is the main Section 5.1 comparison panel requested for the article.
- The figure contains three vertically stacked subplots, one for each constant plate temperature: -10, -15, and -20 deg C.
- Within each subplot, the three thermocouple heights are plotted together: 3.0, 6.2, and 11.0 mm.
- Experimental runs were aligned to the fill/insertion event and interpolated onto a common time base.
- Solid colored curves show experimental means and translucent bands show +/- 1 SD.
- Dashed curves in the same colors show the corresponding `with_rho_temperature_dependent` calibrated simulation.
- Thin gray traces show individual experimental runs for context.
- Line widths are intentionally thin to match the compact comparison-plot style used in the working draft.

## Caption

Experimental and simulated temperature histories for the three constant-plate validation experiments. Each panel corresponds to one plate setpoint and contains the three thermocouple heights. Thin gray traces show individual experimental runs, solid colored curves and shaded bands show the experimental mean +/- 1 SD, and dashed colored curves show the corresponding rho(T) calibrated simulation.

## Interpretation Notes

- This panel is intended to replace the separate per-temperature/per-thermocouple panels in the main text.
- It makes the temperature-setpoint effect visible in one figure while preserving the three thermocouple comparisons inside each condition.
- The largest systematic mismatch remains in the later/upper-probe response, consistent with the limitations of a conduction-dominated model.
