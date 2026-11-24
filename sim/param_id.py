"""Phase 3 dry run: OCV-SOC table and 1-RC equivalent-circuit fit from pulse tests.

The 'true' cell has two RC pairs; the fitted model has one. The fit error is the
model mismatch you will also see on hardware.

    python -m sim.param_id
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .cell import CellParams, Pack, ocv, r0_of

ROOT = Path(__file__).resolve().parents[1]
DT = 0.1
SOC_POINTS = (0.9, 0.7, 0.5, 0.3, 0.1)
I_PULSE, T_PULSE, T_REST = 2.5, 10.0, 600.0       # 1 C for 10 s, then 10 min rest
V_NOISE = 0.5e-3


def pulse_test(cell: CellParams, soc0: float, rng):
    pack = Pack([cell], [soc0])
    n_pre, n_on, n_rest = int(10 / DT), int(T_PULSE / DT), int(T_REST / DT)
    current = np.r_[np.zeros(n_pre), np.full(n_on, I_PULSE), np.zeros(n_rest)]
    v = np.array([pack.step(i, DT)[0] for i in current])
    return current, v + V_NOISE * rng.standard_normal(len(v))


def fit_pulse(current, v):
    on = np.argmax(current > 0)
    off = on + int(T_PULSE / DT)
    v_pre = v[:on].mean()
    r0_on = (v_pre - v[on]) / I_PULSE              # includes 0.1 s of RC: a small bias, as on hardware
    r0_off = (v[off] - v[off - 1]) / I_PULSE
    r0 = 0.5 * (r0_on + r0_off)
    tr = np.arange(len(v) - off - 1) * DT
    vr = v[off + 1:]
    f = lambda t, vinf, a, tau: vinf - a * np.exp(-t / tau)
    (vinf, a, tau), _ = curve_fit(f, tr, vr, p0=(vr[-1], vr[-1] - vr[0], 30.0), maxfev=20000)
    r1 = a / (I_PULSE * (1 - np.exp(-T_PULSE / tau)))
    return dict(r0=r0, r1=r1, tau=tau, c1=tau / r1, v_pre=v_pre, vinf=vinf)


def simulate_1rc(current, v_pre, fit, slope_v_per_ah):
    """Replay the pulse through the fitted model; OCV falls with the charge removed."""
    v1, q, out = 0.0, 0.0, []
    a = np.exp(-DT / fit["tau"])
    for i in current:
        q += i * DT / 3600
        v1 = a * v1 + fit["r1"] * (1 - a) * i
        out.append(v_pre - slope_v_per_ah * q - i * fit["r0"] - v1)
    return np.array(out)


def ocv_table(cell: CellParams, rest_s: float, step=0.1, i_dis=1.25):
    """Discharge in 10 % SOC steps, rest, read voltage. Short rests leave RC voltage behind."""
    pack = Pack([cell], [1.0])
    dt = 1.0
    socs, volts = [1.0], [pack.terminal(0.0)[0]]
    while pack.soc[0] > step + 1e-6:
        target = pack.soc[0] - step
        while pack.soc[0] > target:
            pack.step(i_dis, dt)
        for _ in range(int(rest_s / dt)):
            pack.step(0.0, dt)
        socs.append(pack.soc[0])
        volts.append(pack.terminal(0.0)[0])
    return np.array(socs), np.array(volts)


def identify(cell: CellParams = CellParams(), seed=11, verbose=False):
    """Run the pulse tests and return the fitted ECM lookup table."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in SOC_POINTS:
        current, v = pulse_test(cell, s, rng)
        fit = fit_pulse(current, v)
        slope = (ocv(s) - ocv(s - 0.01)) / (0.01 * cell.q_ah)
        model = simulate_1rc(current, fit["v_pre"], fit, slope)
        rms = float(np.sqrt(np.mean((model - v) ** 2)))
        rows.append(dict(soc=s, r0_fit=fit["r0"], r0_true=float(r0_of(cell, s)), r1_fit=fit["r1"],
                         tau_fit=fit["tau"], c1_fit=fit["c1"], replay_rms_mv=1e3 * rms,
                         _trace=(current, v, model)))
    return rows


def ecm_lookup(rows):
    socs = np.array([r["soc"] for r in rows])[::-1]
    r0 = np.array([r["r0_fit"] for r in rows])[::-1]
    return dict(soc=socs, r0=r0, r1=float(np.mean([r["r1_fit"] for r in rows])),
                tau=float(np.mean([r["tau_fit"] for r in rows])))


def main():
    cell = CellParams()
    rows = identify(cell)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open(ROOT / "data" / "ecm_fit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)

    print("Pulse-test fit (1 C, 10 s pulse, 10 min rest)")
    print(f"  {'SOC':>4s} {'R0 fit':>8s} {'R0 true':>8s} {'R1 fit':>8s} {'tau fit':>8s} {'C1 fit':>8s} {'replay RMS':>11s}")
    for r in rows:
        print(f"  {r['soc']:4.1f} {r['r0_fit'] * 1e3:6.1f}mO {r['r0_true'] * 1e3:6.1f}mO {r['r1_fit'] * 1e3:6.1f}mO "
              f"{r['tau_fit']:7.1f}s {r['c1_fit']:7.0f}F {r['replay_rms_mv']:8.2f} mV")
    print("  True cell: R1 15 mOhm tau 30 s, plus a slow RC of 8 mOhm, tau 400 s that the 1-RC model cannot hold.")

    print("\nOCV table error vs the true curve, by rest time after each 10 % step")
    tables = {}
    for rest in (600, 3600):
        s, v = ocv_table(cell, rest)
        err = 1e3 * (v - ocv(s))
        tables[rest] = (s, v)
        print(f"  {rest // 60:3d} min rest: mean {err.mean():+.1f} mV, worst {err[np.argmax(np.abs(err))]:+.1f} mV")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    r = rows[2]
    current, v, model = r["_trace"]
    t = np.arange(len(v)) * DT
    axes[0].plot(t, v, color="#2a6fdb", lw=1, label="Measured (true 2-RC cell + noise)")
    axes[0].plot(t, model, color="#e07b00", lw=1, ls="--", label=f"Fitted 1-RC model ({r['replay_rms_mv']:.1f} mV RMS)")
    axes[0].set_xlim(0, 300)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Cell voltage (V)")
    axes[0].set_title("Pulse test at 50 % SOC: 2.5 A for 10 s", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    fine = np.linspace(0, 1, 200)
    axes[1].plot(fine * 100, ocv(fine), color="0.4", lw=1, label="True OCV curve")
    for rest, c in ((600, "#c43f3f"), (3600, "#1a9e5c")):
        s, vv = tables[rest]
        axes[1].plot(s * 100, vv, "o", ms=4, color=c, label=f"Measured after {rest // 60} min rest")
    axes[1].set_xlabel("SOC (%)")
    axes[1].set_ylabel("Volts")
    axes[1].set_title("OCV table: short rests read low", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "plots" / "ecm_fit.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote data/ecm_fit.csv and plots/ecm_fit.png under {ROOT}")


if __name__ == "__main__":
    main()
