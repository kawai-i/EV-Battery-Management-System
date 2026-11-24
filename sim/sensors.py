"""Phase 1 dry run: sensor error models, calibration, and the divider-drain problem.

Option B front end: each cell tap goes through a resistor divider to an ADS1115.
Cell voltage = difference of two calibrated tap readings.

    python -m sim.sensors
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LSB = 4.096 / 32768          # ADS1115 at +/-4.096 V full scale


class TapFrontEnd:
    """Resistor dividers (0.1 %) + ADC gain/offset error + noise + quantisation, per tap."""

    def __init__(self, n=4, rng=None, r_tol=0.001, gain_sigma=0.0015, offset_sigma=1e-3,
                 noise=0.3e-3, r_bottom=33e3):
        rng = np.random.default_rng(1) if rng is None else rng
        k = np.arange(1, n + 1)
        self.r_nom = 0.9 * 4.096 / (4.2 * k)                        # top tap maps to 90 % of FS
        r2 = r_bottom * (1 + r_tol * rng.uniform(-1, 1, n))
        r1 = r_bottom * (1 / self.r_nom - 1) * (1 + r_tol * rng.uniform(-1, 1, n))
        self.r_act = r2 / (r1 + r2)
        self.r_total = r_bottom / self.r_nom                        # divider resistance per tap
        self.gain = 1 + gain_sigma * rng.standard_normal(n)
        self.offset = offset_sigma * rng.standard_normal(n)
        self.noise = noise
        self.cal = np.tile([1.0, 0.0], (n, 1))                      # tap = a * raw + b

    def adc(self, taps, rng):
        taps = np.atleast_2d(taps)
        v = taps * self.r_act * self.gain + self.offset + self.noise * rng.standard_normal(taps.shape)
        return np.round(v / LSB) * LSB

    def taps_raw(self, adc_v):
        return adc_v / self.r_nom

    def taps(self, adc_v):
        return self.cal[:, 0] * self.taps_raw(adc_v) + self.cal[:, 1]

    def calibrate(self, cell_points, rng, samples=16):
        """Cell simulator sets all cells to each point; fit each tap against the reference meter."""
        n = len(self.r_nom)
        for j in range(n):
            ref, raw = [], []
            for v in cell_points:
                taps = np.full(n, v) * np.arange(1, n + 1)
                raw.append(self.taps_raw(self.adc(np.tile(taps, (samples, 1)), rng)).mean(0)[j])
                ref.append(taps[j])
            self.cal[j] = np.polyfit(raw, ref, 1)


def cell_voltages(taps):
    taps = np.atleast_2d(taps)
    return np.diff(np.concatenate([np.zeros((len(taps), 1)), taps], axis=1), axis=1)


class CurrentSensor:
    """INA226 + 10 mOhm shunt: gain error from shunt tolerance, offset from IC and layout."""

    def __init__(self, gain_err=0.008, offset=0.018, noise=0.002):
        self.gain_err, self.offset, self.noise = gain_err, offset, noise
        self.cal = np.array([1.0, 0.0])

    def raw(self, i, rng):
        i = np.asarray(i, dtype=float)
        return i * (1 + self.gain_err) + self.offset + self.noise * rng.standard_normal(i.shape)

    def read(self, i, rng):
        return self.cal[0] * self.raw(i, rng) + self.cal[1]

    def calibrate(self, points, rng, samples=50):
        raw = [self.raw(np.full(samples, p), rng).mean() for p in points]
        self.cal = np.polyfit(raw, points, 1)


class NTC:
    """10 k NTC with a 10 k pull-up. Part tolerance: R25 +/-1 %, Beta +/-1 %."""

    def __init__(self, r25_err=0.01, beta_err=0.01, beta=3950.0):
        self.beta_nom, self.beta = beta, beta * (1 + beta_err)
        self.r25 = 10e3 * (1 + r25_err)
        self.cal = np.array([1.0, 0.0])

    def raw(self, t_c, rng):
        tk = np.asarray(t_c) + 273.15
        r = self.r25 * np.exp(self.beta * (1 / tk - 1 / 298.15))
        r = r * (1 + 0.0005 * rng.standard_normal(np.shape(r)))
        return 1 / (1 / 298.15 + np.log(r / 10e3) / self.beta_nom) - 273.15

    def read(self, t_c, rng):
        return self.cal[0] * self.raw(t_c, rng) + self.cal[1]

    def calibrate(self, points, rng):
        raw = [self.raw(np.full(20, p), rng).mean() for p in points]
        self.cal = np.polyfit(raw, points, 1)


def divider_drain(fe: TapFrontEnd, v_cell=3.7, duty=1.0):
    """Per-cell rest current drawn by the dividers. Tap k's divider loads cells 1..k."""
    n = len(fe.r_nom)
    taps = v_cell * np.arange(1, n + 1)
    i_tap = taps / fe.r_total * duty
    return np.array([i_tap[j:].sum() for j in range(n)])


def main():
    rng = np.random.default_rng(2026)
    rows = []

    fe = TapFrontEnd(rng=rng)
    test = rng.uniform(3.0, 4.2, (400, 4))
    taps_true = np.cumsum(test, axis=1)
    adc = fe.adc(taps_true, rng)
    before = cell_voltages(fe.taps_raw(adc)) - test
    fe.calibrate([3.0, 3.3, 3.6, 3.9, 4.2], rng)
    adc = fe.adc(taps_true, rng)
    after = cell_voltages(fe.taps(adc)) - test
    for j in range(4):
        rows.append(dict(channel=f"Cell {j + 1} voltage", unit="mV",
                         rms_before=1e3 * np.sqrt(np.mean(before[:, j] ** 2)), max_before=1e3 * np.abs(before[:, j]).max(),
                         rms_after=1e3 * np.sqrt(np.mean(after[:, j] ** 2)), max_after=1e3 * np.abs(after[:, j]).max()))

    cs = CurrentSensor()
    i_test = rng.uniform(-2.5, 5, 400)
    e0 = cs.read(i_test, rng) - i_test
    cs.calibrate([-2, -1, -0.5, 0, 0.5, 1, 2], rng)
    e1 = cs.read(i_test, rng) - i_test
    rows.append(dict(channel="Pack current", unit="mA", rms_before=1e3 * np.sqrt(np.mean(e0 ** 2)),
                     max_before=1e3 * np.abs(e0).max(), rms_after=1e3 * np.sqrt(np.mean(e1 ** 2)),
                     max_after=1e3 * np.abs(e1).max()))

    ntc = NTC()
    t_test = rng.uniform(10, 50, 400)
    e0 = ntc.read(t_test, rng) - t_test
    ntc.calibrate([15, 30, 45], rng)
    e1 = ntc.read(t_test, rng) - t_test
    rows.append(dict(channel="NTC temperature", unit="degC", rms_before=np.sqrt(np.mean(e0 ** 2)),
                     max_before=np.abs(e0).max(), rms_after=np.sqrt(np.mean(e1 ** 2)), max_after=np.abs(e1).max()))

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    with open(ROOT / "data" / "calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("Calibration (held-out test points)")
    print(f"  {'Channel':18s} {'unit':5s} {'RMS before':>10s} {'max before':>10s} {'RMS after':>10s} {'max after':>10s}")
    for r in rows:
        print(f"  {r['channel']:18s} {r['unit']:5s} {r['rms_before']:10.2f} {r['max_before']:10.2f} "
              f"{r['rms_after']:10.2f} {r['max_after']:10.2f}")

    print("\nDivider drain at rest (3.7 V per cell, 2.5 Ah cells)")
    drains = []
    for label, duty in (("always on", 1.0), ("switched, 2 % duty", 0.02)):
        d = divider_drain(fe, duty=duty)
        spread_mah_day = (d.max() - d.min()) * 24 * 1e3
        drains.append((label, d, spread_mah_day))
        print(f"  {label:20s} per-cell drain uA: {np.round(d * 1e6, 1)}  "
              f"imbalance growth {spread_mah_day:.2f} mAh/day = {spread_mah_day / 2500 * 100 * 30:.2f} % SOC per 30 days")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(4)
    axes[0].bar(x - 0.2, [r["rms_before"] for r in rows[:4]], 0.4, label="before calibration", color="#c43f3f")
    axes[0].bar(x + 0.2, [r["rms_after"] for r in rows[:4]], 0.4, label="after calibration", color="#2a6fdb")
    axes[0].set_xticks(x, [f"Cell {j + 1}" for j in range(4)])
    axes[0].set_ylabel("RMS error (mV)")
    axes[0].set_title("Cell voltage error, dividers + ADS1115")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    for (label, d, _), c, off in zip(drains, ("#c43f3f", "#2a6fdb"), (-0.2, 0.2)):
        axes[1].bar(x + off, d * 1e6, 0.4, label=label, color=c)
    axes[1].set_xticks(x, [f"Cell {j + 1}" for j in range(4)])
    axes[1].set_ylabel("Rest drain (uA)")
    axes[1].set_title("Dividers drain the bottom cell most")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "plots" / "calibration.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote data/calibration.csv and plots/calibration.png under {ROOT}")


if __name__ == "__main__":
    main()
