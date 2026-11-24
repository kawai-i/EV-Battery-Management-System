"""Gate checks for the BMS simulation code. Run from ev-bms/:  python -m pytest -q"""
import numpy as np
import pytest

from sim.cell import CellParams, Pack, docv, ocv, soc_from_ocv
from sim.param_id import ecm_lookup, identify
from sim.protection import FAULT, Limits, Protection, hw_short_circuit
from sim.protection_sim import SCENARIOS, chatter, run
from sim.sensors import CurrentSensor, TapFrontEnd, cell_voltages, divider_drain
from sim.soc import EKF, CoulombCounter

L = Limits()
DELAYS = {"OV": L.ov_delay, "UV": L.uv_delay, "OCD": L.ocd_delay, "OCC": L.occ_delay, "OTD": L.t_delay,
          "OTC": L.t_delay, "UTC": L.t_delay, "SENSOR": L.sensor_delay}


def test_ocv_inverse_and_slope():
    s = np.linspace(0.01, 0.99, 50)
    assert np.allclose(soc_from_ocv(ocv(s)), s, atol=1e-9)
    assert np.all(docv(s) > 0)


def test_coulomb_counting_matches_plant():
    pack = Pack([CellParams()], [0.9])
    cc = CoulombCounter(0.9, 2.5)
    for _ in range(3600):
        pack.step(1.25, 1.0)
        cc.update(1.25, 0.0, 1.0)
    assert abs(pack.soc[0] - 0.4) < 1e-9 and abs(cc.soc - 0.4) < 1e-9


def test_normal_operation_never_trips():
    rng = np.random.default_rng(0)
    p = Protection()
    for k in range(3000):
        p.update(3.7 + 0.001 * rng.standard_normal(4), 1.5 * np.sin(k / 50), 30 + 0.1 * rng.standard_normal(4))
        assert p.state != FAULT


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_fault_trips_after_its_debounce(name):
    r, _ = run(name)
    assert r["response_ms"] is not None, "did not trip"
    fault = r["fault"]
    # Measured from the true crossing; sensor noise may start the timer a few ticks early or late.
    assert DELAYS[fault] * 1000 - 300 <= r["response_ms"] <= DELAYS[fault] * 1000 + 700
    assert r["reset_blocked_while_present"] and r["reset_ok_after_clear"]


def test_debounce_and_hysteresis_stop_chatter():
    _, out = chatter()
    assert out["Debounce + hysteresis"][1] <= 2
    assert out["No debounce, no hysteresis"][1] > 50


def test_hardware_short_circuit_under_1ms():
    t_off, _ = hw_short_circuit()
    assert t_off < 1e-3


def test_calibration_reduces_errors():
    rng = np.random.default_rng(1)
    fe = TapFrontEnd(rng=rng)
    cells = rng.uniform(3.0, 4.2, (200, 4))
    taps = np.cumsum(cells, axis=1)
    before = np.abs(cell_voltages(fe.taps_raw(fe.adc(taps, rng))) - cells).max()
    fe.calibrate([3.0, 3.3, 3.6, 3.9, 4.2], rng)
    after = np.abs(cell_voltages(fe.taps(fe.adc(taps, rng))) - cells).max()
    assert after < before and after < 0.010
    cs = CurrentSensor()
    cs.calibrate([-2, -1, 0, 1, 2], rng)
    assert abs(float(np.mean(cs.read(np.zeros(500), rng)))) < 0.002


def test_switched_dividers_cut_drain_imbalance():
    fe = TapFrontEnd()
    on, sw = divider_drain(fe), divider_drain(fe, duty=0.02)
    assert on[0] > on[-1]                                # bottom cell drains most
    assert (sw.max() - sw.min()) < 0.05 * (on.max() - on.min())


def test_ecm_fit_close_to_truth():
    rows = identify()
    for r in rows:
        assert abs(r["r0_fit"] - r["r0_true"]) / r["r0_true"] < 0.05
        assert r["replay_rms_mv"] < 10                   # Gate 4


def test_ekf_recovers_from_wrong_start():
    ecm = ecm_lookup(identify())
    pack = Pack([CellParams()], [0.8])
    ekf = EKF(0.5, 2.5, ecm)
    rng = np.random.default_rng(2)
    for k in range(1800):
        i = 2.0 if (k // 60) % 2 == 0 else 0.3
        v = pack.step(i, 1.0)[0] + 0.001 * rng.standard_normal()
        ekf.update(i, v, 1.0)
    assert abs(ekf.soc - pack.soc[0]) < 0.03
