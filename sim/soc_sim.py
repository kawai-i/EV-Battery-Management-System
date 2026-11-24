"""Phase 4 dry run: SOC estimators on a dynamic drive profile, scored against the true SOC.

Plant    : 4 cells with capacity and resistance spread, 2-RC dynamics, R0 rising at low SOC.
Sensors  : the calibrated divider + ADS1115 front end and INA226 models from sim.sensors.
Profile  : pulsed drive blocks (including regen), 20-minute parking rests every 45 minutes,
           until the weakest cell nears its minimum voltage.
Scenario A: estimators start from the right SOC.
Scenario B: estimators start 30 % low (for example after a reset during a drive).

    python -m sim.soc_sim
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cell import Pack, make_cells
from .param_id import ecm_lookup, identify
from .sensors import CurrentSensor, TapFrontEnd, cell_voltages
from .soc import EKF, CoulombCounter, OcvAnchoredCC

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0
BLOCK = [(2.0, 60), (0.5, 120), (4.0, 20), (0.0, 60), (-1.0, 15), (1.5, 90), (3.0, 30), (0.0, 90), (0.8, 120)]
BLOCK_T = np.cumsum([d for _, d in BLOCK])
DRIVE_S, PARK_S = 2700, 1200


def profile_current(t):
    c = t % (DRIVE_S + PARK_S)
    if c >= DRIVE_S:
        return 0.0
    k = int(np.searchsorted(BLOCK_T, c % BLOCK_T[-1], side="right"))
    return BLOCK[k][0]


def run(offset, seed=5):
    rng = np.random.default_rng(seed)
    cells = make_cells(4, np.random.default_rng(42))
    soc0 = np.array([0.95, 0.94, 0.96, 0.95])
    pack = Pack(cells, soc0)
    q_meas = np.array([c.q_ah for c in cells]) * (1 + 0.005 * rng.standard_normal(4))   # Phase 3 capacity test
    ecm = ecm_lookup(identify())

    fe = TapFrontEnd(rng=np.random.default_rng(1))
    fe.calibrate([3.0, 3.3, 3.6, 3.9, 4.2], rng)
    cs_cal, cs_raw = CurrentSensor(), CurrentSensor()
    cs_cal.calibrate([-2, -1, -0.5, 0, 0.5, 1, 2], rng)

    start = np.clip(soc0 + offset, 0, 1)
    methods = {
        "Coulomb counting, uncalibrated current": [CoulombCounter(s, q) for s, q in zip(start, q_meas)],
        "Coulomb counting": [CoulombCounter(s, q) for s, q in zip(start, q_meas)],
        "OCV + Coulomb counting": [OcvAnchoredCC(s, q) for s, q in zip(start, q_meas)],
        "EKF (1-RC)": [EKF(s, q, ecm) for s, q in zip(start, q_meas)],
    }
    t, log_t, log_true, log_est, log_i = 0.0, [], [], {m: [] for m in methods}, []
    while True:
        i = profile_current(t)
        v_true = pack.step(i, DT)
        if v_true.min() < 3.05 or pack.soc.min() < 0.03:
            break
        v_meas = cell_voltages(fe.taps(fe.adc(np.cumsum(v_true), rng)))[0]
        i_cal = float(cs_cal.read(i, rng))
        i_raw = float(cs_raw.read(i, rng))
        for name, ests in methods.items():
            ii = i_raw if "uncalibrated" in name else i_cal
            log_est[name].append([e.update(ii, float(v), DT) for e, v in zip(ests, v_meas)])
        log_t.append(t)
        log_true.append(pack.soc.copy())
        log_i.append(i)
        t += DT
    return np.array(log_t), np.array(log_true), {k: np.array(v) for k, v in log_est.items()}, np.array(log_i)


def converge_time(t, err, tol=0.03):
    """Time after which |error| stays below tol on every cell."""
    bad = np.any(np.abs(err) >= tol, axis=1)
    if not bad.any():
        return 0.0
    last = np.where(bad)[0][-1]
    return None if last == len(t) - 1 else float(t[last + 1])


def main():
    rows, runs = [], {}
    for label, offset in (("A: right start", 0.0), ("B: 30 % low start", -0.30)):
        t, true, est, cur = run(offset)
        runs[label] = (t, true, est, cur)
        for name, e in est.items():
            err = e - true
            ct = converge_time(t, err)
            rows.append(dict(scenario=label, method=name, rms_pct=100 * np.sqrt(np.mean(err ** 2)),
                             max_pct=100 * np.abs(err).max(), final_pct=100 * np.abs(err[-1]).max(),
                             converge_min=None if ct is None else ct / 60))
        print(f"Scenario {label}: {t[-1] / 3600:.2f} h profile, {true[0].mean() * 100:.0f} % to "
              f"{true[-1].min() * 100:.0f} % SOC (weakest cell)")

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    with open(ROOT / "data" / "soc_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'Scenario':18s} {'Method':40s} {'RMS':>6s} {'max':>6s} {'final':>6s} {'within 3% after':>16s}")
    for r in rows:
        ct = "never" if r["converge_min"] is None else (
            f"{r['converge_min'] * 60:.0f} s" if r["converge_min"] < 1 else f"{r['converge_min']:.0f} min")
        print(f"{r['scenario']:18s} {r['method']:40s} {r['rms_pct']:5.2f}% {r['max_pct']:5.1f}% "
              f"{r['final_pct']:5.2f}% {ct:>16s}")

    colors = {"Coulomb counting, uncalibrated current": "#c43f3f", "Coulomb counting": "#e07b00",
              "OCV + Coulomb counting": "#2a6fdb", "EKF (1-RC)": "#1a9e5c"}
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 2.2, 2.2]))
    t, true, est, cur = runs["A: right start"]
    axes[0].plot(t / 60, cur, color="0.35", lw=0.5)
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title("Dynamic profile: pulsed drive with regen, 20-minute parking rests", fontsize=10)
    axes[0].grid(alpha=0.3)
    for ax, label in ((axes[1], "A: right start"), (axes[2], "B: 30 % low start")):
        t, true, est, _ = runs[label]
        for name, e in est.items():
            ax.plot(t / 60, 100 * (e[:, 0] - true[:, 0]), color=colors[name], lw=1, label=name)
        ax.axhline(0, color="0.3", lw=0.6)
        ax.axhspan(-3, 3, color="0.85", alpha=0.5, lw=0)
        ax.set_ylabel("SOC error, cell 1 (%)")
        ax.set_title(f"Scenario {label} (grey band = ±3 % target)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[1].set_ylim(-6, 6)
    axes[2].set_ylim(-35, 8)
    axes[1].legend(fontsize=8, loc="lower left")
    axes[2].set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(ROOT / "plots" / "soc_errors.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote data/soc_errors.csv and plots/soc_errors.png under {ROOT}")


if __name__ == "__main__":
    main()
