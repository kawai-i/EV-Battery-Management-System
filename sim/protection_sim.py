"""Phase 2 dry run: fault injection against the protection state machine.

Each scenario drives the sensor readings the way the bench test will (cell simulator,
electronic load, hair dryer, unplugged NTC, frozen ADC) and logs the response time from
the moment the true condition starts to the moment the MOSFETs open.

    python -m sim.protection_sim
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .protection import FAULT, Limits, Protection, hw_short_circuit

ROOT = Path(__file__).resolve().parents[1]
DT = 0.1
L = Limits()


def scenario(name, n=1500, seed=0):
    """Return true signals (v[n,4], i[n], t[n,4], age[n]) and the index where the fault begins."""
    rng = np.random.default_rng(seed)
    k = np.arange(n) * DT
    v = np.full((n, 4), 3.70)
    i = np.full(n, 1.0)
    t = np.full((n, 4), 30.0)
    age = np.zeros(n)
    if name == "OV (cell simulator ramp, 5 mV/s)":
        v[:, 1] = 4.15 + 0.005 * k
        start = np.argmax(v[:, 1] > L.ov_trip)
    elif name == "UV (cell simulator ramp, 5 mV/s)":
        v[:, 2] = 3.05 - 0.005 * k
        start = np.argmax(v[:, 2] < L.uv_trip)
    elif name == "OCD (load step 1 A to 7 A)":
        start = 100
        i[start:] = 7.0
    elif name == "OCC (charge step to 3.5 A)":
        start = 100
        i[:] = -1.0
        i[start:] = -3.5
    elif name == "OTD (hair dryer, 0.5 degC/s)":
        t[:, 0] = 30 + 0.5 * k
        start = np.argmax(t[:, 0] > L.otd_trip)
    elif name == "OTC (warm while charging)":
        i[:] = -1.0
        t[:, 3] = 35 + 0.25 * k
        start = np.argmax(t[:, 3] > L.otc_trip)
    elif name == "UTC (cold while charging)":
        i[:] = -1.0
        t[:, :] = 5 - 0.1 * k[:, None]
        start = np.argmax(t[:, 0] < L.ut_c_trip)
    elif name == "Sensor: open NTC":
        start = 100
        t[start:, 2] = -45.0
    elif name == "Sensor: ADC frozen":
        start = 100
        age[start:] = (k[start:] - k[start]) + DT
    else:
        raise ValueError(name)
    noisy_v = v + 0.001 * rng.standard_normal(v.shape)
    noisy_i = i + 0.005 * rng.standard_normal(n)
    noisy_t = t + 0.1 * rng.standard_normal(t.shape)
    return noisy_v, noisy_i, noisy_t, age, int(start)


SCENARIOS = ["OV (cell simulator ramp, 5 mV/s)", "UV (cell simulator ramp, 5 mV/s)",
             "OCD (load step 1 A to 7 A)", "OCC (charge step to 3.5 A)",
             "OTD (hair dryer, 0.5 degC/s)", "OTC (warm while charging)", "UTC (cold while charging)",
             "Sensor: open NTC", "Sensor: ADC frozen"]


def run(name):
    v, i, t, age, start = scenario(name)
    p = Protection(dt=DT)
    trip, fault = None, None
    for k in range(len(i)):
        p.update(v[k], i[k], t[k], age[k])
        if p.state == FAULT and trip is None:
            trip, fault = k, sorted(p.active)
    # Reset test: try straight away (condition still present), then with the bench restored.
    blocked = not p.reset(v[-1], i[-1], t[-1], age[-1])
    ok = p.reset(np.full(4, 3.7), 0.0, np.full(4, 25.0))
    return dict(scenario=name, fault=",".join(fault or []),
                response_ms=None if trip is None else round((trip - start) * DT * 1000),
                reset_blocked_while_present=blocked, reset_ok_after_clear=ok), (v, i, t, start, trip)


def chatter():
    """Cell voltage hovering at the OV threshold with 3 mV noise: count MOSFET toggles."""
    rng = np.random.default_rng(3)
    n = 1200
    v = np.full((n, 4), 3.9)
    v[:, 0] = 4.198 + 0.004 * np.sin(np.arange(n) * DT * 0.5) + 0.003 * rng.standard_normal(n)
    out = {}
    for label, kw in (("No debounce, no hysteresis", dict(use_debounce=False, use_hysteresis=False)),
                      ("Debounce + hysteresis", dict())):
        p = Protection(dt=DT, latch=False, **kw)
        fets = []
        for k in range(n):
            p.update(v[k], -0.5, np.full(4, 25.0))
            fets.append(p.fets_on)
        fets = np.array(fets)
        out[label] = (fets, int(np.sum(fets[1:] != fets[:-1])))
    return v[:, 0], out


def main():
    rows, traces = [], {}
    for s in SCENARIOS:
        r, tr = run(s)
        rows.append(r)
        traces[s] = tr

    t_sc, i2t_hw = hw_short_circuit()
    # Firmware-only alternative: seen at the next 10 Hz tick, then the OCD debounce.
    t_fw = 0.1 + L.ocd_delay
    i2t_fw = 100.0 ** 2 * t_fw
    rows.append(dict(scenario="Short circuit, hardware comparator", fault="SC (hardware)",
                     response_ms=round(t_sc * 1000, 3), reset_blocked_while_present=True, reset_ok_after_clear=True))

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    with open(ROOT / "data" / "protection_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'Scenario':38s} {'Fault':9s} {'Response':>10s}  Reset blocked / ok")
    for r in rows:
        resp = "NO TRIP" if r["response_ms"] is None else f"{r['response_ms']} ms"
        print(f"{r['scenario']:38s} {r['fault']:9s} {resp:>10s}  {r['reset_blocked_while_present']} / {r['reset_ok_after_clear']}")
    print(f"\nShort circuit (100 A prospective): hardware trip {t_sc * 1e3:.3f} ms, I2t {i2t_hw:.2f} A2s; "
          f"firmware-only path up to {t_fw * 1e3:.0f} ms, I2t {i2t_fw:.0f} A2s ({i2t_fw / i2t_hw:.0f}x more)")

    vc, ch = chatter()
    for label, (_, toggles) in ch.items():
        print(f"Chatter test at the OV threshold, {label}: {toggles} MOSFET toggles in {len(vc) * DT:.0f} s")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=False)
    v, i, t, start, trip = traces[SCENARIOS[0]]
    tt = np.arange(len(i)) * DT
    ax = axes[0]
    ax.plot(tt, v[:, 1], color="#2a6fdb", lw=1, label="Cell 2 voltage (cell simulator ramp)")
    ax.axhline(L.ov_trip, color="#c43f3f", ls="--", lw=1, label="OV trip 4.20 V")
    ax.axhline(L.ov_clear, color="0.5", ls=":", lw=1, label="OV clear 4.10 V (hysteresis)")
    ax.axvline(start * DT, color="0.3", lw=0.8)
    ax.axvline(trip * DT, color="#c43f3f", lw=0.8)
    ax.annotate(f"trip after {(trip - start) * DT * 1000:.0f} ms (0.5 s debounce)", (trip * DT, 4.21),
                xytext=(trip * DT + 3, 4.235), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlim(start * DT - 8, start * DT + 12)
    ax.set_ylim(4.14, 4.26)
    ax.set_ylabel("Volts")
    ax.set_title("Fault injection: overvoltage trips once, after the debounce", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax = axes[1]
    tc = np.arange(len(vc)) * DT
    ax.plot(tc, vc, color="#2a6fdb", lw=0.7, label="Cell 1 voltage at the threshold")
    ax.axhline(L.ov_trip, color="#c43f3f", ls="--", lw=1)
    for (label, (fets, toggles)), off, c in zip(ch.items(), (4.170, 4.160), ("#c43f3f", "#1a9e5c")):
        ax.step(tc, off + 0.006 * fets, where="post", color=c, lw=1, label=f"MOSFETs on: {label} ({toggles} toggles)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Volts / FET state")
    ax.set_title("Chatter: debounce and hysteresis stop the MOSFETs toggling at the threshold", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "plots" / "protection.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote data/protection_tests.csv and plots/protection.png under {ROOT}")


if __name__ == "__main__":
    main()
