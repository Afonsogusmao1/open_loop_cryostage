# fig_5_1_zero_crossing_times

## Files

- `fig_5_1_zero_crossing_times.png`
- `fig_5_1_zero_crossing_times.pdf`
- `section_5_1_zero_crossing_times.csv`

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

## Method

- For each experimental run and simulated probe trace, the reported time is the first downward crossing of 0 deg C after the fill/insertion event.
- Crossing times were linearly interpolated between neighboring samples.
- Bars show mean experimental crossing time for n = 5 runs, error bars show sample SD, black circles show the five individual runs, and black diamonds show the simulation.
- Probe colors match the temperature-history panels: 3.0 mm teal, 6.2 mm blue, and 11.0 mm orange.

## Values

| Plate setpoint | Probe | Experimental mean +/- SD (s) | Simulation (s) | Simulation - experiment (s) |
|---|---:|---:|---:|---:|
| -10 deg C | 3.0 mm | 76.8 +/- 6.1 | 75.7 | -1.1 |
| -10 deg C | 6.2 mm | 357.2 +/- 28.8 | 345.0 | -12.2 |
| -10 deg C | 11.0 mm | 967.4 +/- 43.6 | 1018.6 | +51.2 |
| -15 deg C | 3.0 mm | 44.7 +/- 1.5 | 46.9 | +2.2 |
| -15 deg C | 6.2 mm | 187.2 +/- 8.7 | 217.1 | +29.9 |
| -15 deg C | 11.0 mm | 614.4 +/- 28.7 | 669.3 | +54.8 |
| -20 deg C | 3.0 mm | 34.6 +/- 3.0 | 34.6 | +0.1 |
| -20 deg C | 6.2 mm | 143.1 +/- 12.3 | 155.7 | +12.7 |
| -20 deg C | 11.0 mm | 428.7 +/- 23.9 | 494.2 | +65.5 |

## Caption

Comparison of experimentally inferred and simulated 0 deg C crossing times for the constant-temperature validation experiments. Bars show the mean of five experimental runs, error bars show 1 SD, black circles show individual runs, and black diamonds show the corresponding rho(T) simulation.

## Interpretation Notes

- Agreement is strongest close to the cold plate, where the imposed thermal boundary dominates.
- The simulated crossing tends to occur later than the experimental mean at the upper probes, particularly at 11.0 mm.
- This pattern is consistent with a purely conductive model that does not explicitly resolve liquid-phase natural convection, mixing during and after filling, or local disturbances around thermocouple junctions.
