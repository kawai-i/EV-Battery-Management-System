# EV BMS: software-in-the-loop simulations

Pre-hardware simulations for the EV Battery Management System upgrade plan (v2). They run on a
laptop with Python 3, NumPy, SciPy and Matplotlib. No cells are needed.

The protection state machine (`sim/protection.py`) and SOC estimators (`sim/soc.py`) are written as
firmware twins: one `update()` per tick, plain numbers in and out, so they port line by line to C.

| Script | Plan phase | What it shows | Outputs |
| --- | --- | --- | --- |
| `python -m sim.sensors` | Phase 1 | Calibration of divider + ADS1115 cell channels, INA226 current, NTCs; divider drain imbalance | `data/calibration.csv`, `plots/calibration.png` |
| `python -m sim.protection_sim` | Phase 2 | Fault injection for every protection, reset rules, chatter test, hardware short-circuit trip | `data/protection_tests.csv`, `plots/protection.png` |
| `python -m sim.param_id` | Phase 3 | Pulse-test fit of R0, R1, C1 at 5 SOC points; OCV table vs rest time | `data/ecm_fit.csv`, `plots/ecm_fit.png` |
| `python -m sim.soc_sim` | Phase 4 | Coulomb counting, OCV + CC and EKF on a dynamic profile, right start and 30 % wrong start | `data/soc_errors.csv`, `plots/soc_errors.png` |
| `python -m sim.balancing_sim` | Phase 5 | Passive top balancing on vs off over 4 cycles | `data/balancing.csv`, `plots/balancing.png` |
| `python -m pytest -q` | all | Gate checks | |

## The plant model

- 4 cells from one batch: about 2 % capacity spread and 8 % resistance spread
- the true cell has OCV(SOC), R0 rising at low SOC, a fast RC (30 s) **and** a slow RC (400 s)
- the estimators only know a 1-RC model fitted from pulse tests, so the model mismatch is real
- the OCV curve is a representative NMC 18650 curve: replace `SOC_PTS` / `OCV_PTS` in
  `sim/cell.py` with your Phase 3 measurements

Positive current = discharge, everywhere.
