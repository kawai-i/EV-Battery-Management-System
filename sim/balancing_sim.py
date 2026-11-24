"""Phase 5 dry run: passive top balancing over repeated charge / discharge cycles.

Cell 3 starts 100 mAh low. Each cycle: CC charge at 1.25 A, cell-level CV at 4.19 V until
the current falls below C/20, 1 h rest, 2 A discharge to 3.0 V on the weakest cell, 30 min rest.
Balancing (when on): bleed any cell more than 20 mV above the lowest (off again below 10 mV)
while the highest cell is above 4.0 V, through 60 ohm (about 70 mA at 4.2 V).

    python -m sim.balancing_sim
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cell import Pack, make_cells, ocv, r0_of

ROOT = Path(__file__).resolve().parents[1]
DT = 2.0
R_BLEED, V_ON, V_OFF, V_TOP = 60.0, 0.020, 0.010, 4.0
R_TH_RESISTOR = 100.0      # K/W, small through-hole resistor on the board


def simulate(balance: bool, cycles: int = 4):
    cells = make_cells(4, np.random.default_rng(42), q_spread=0.0)   # same capacity: isolate SOC imbalance
    soc0 = np.array([0.30, 0.30, 0.30 - 0.100 / 2.5, 0.30])          # cell 3 starts 100 mAh low
    pack = Pack(cells, soc0)
    bleeding = np.zeros(4, bool)
    t, log, eoc, delivered = 0.0, [], [], []
    v = pack.terminal(0.0)

    def tick(i_pack):
        nonlocal t, v, bleeding
        if balance:
            spread = v - v.min()
            top = v.max() > V_TOP
            bleeding = top & np.where(bleeding, spread > V_OFF, spread > V_ON)
        i_bleed = np.where(bleeding, v / R_BLEED, 0.0)
        v = pack.step(i_pack, DT, i_extra=i_bleed)
        t += DT
        log.append((t, *v, *bleeding, i_pack))
        return v

    for _ in range(cycles):
        i = -1.25                                     # CC charge
        while v.max() < 4.19:
            tick(i)
        while True:                                   # CV: hold the highest cell at 4.19 V
            r0 = np.array([r0_of(c, s) for c, s in zip(cells, pack.soc)])
            i_cell_limit = (ocv(pack.soc) - pack.v1 - pack.v2 - 4.19) / r0   # most negative current each cell allows
            i = float(np.clip(i_cell_limit.max(), -1.25, 0.0))
            if -i < 0.125:
                break
            tick(i)
        for _ in range(int(3600 / DT)):               # rest (balancing continues)
            tick(0.0)
        eoc.append(1e3 * (v.max() - v.min()))
        ah = 0.0
        while v.min() > 3.0:                          # discharge
            tick(2.0)
            ah += 2.0 * DT / 3600
        delivered.append(ah)
        for _ in range(int(1800 / DT)):
            tick(0.0)
    return np.array(log), np.array(eoc), np.array(delivered)


def main():
    res = {}
    for label, on in (("Balancing off", False), ("Balancing on", True)):
        res[label] = simulate(on)

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    with open(ROOT / "data" / "balancing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "cycle", "spread_end_of_charge_mV", "delivered_Ah"])
        for label, (_, eoc, ah) in res.items():
            for k, (s, a) in enumerate(zip(eoc, ah), 1):
                w.writerow([label, k, round(s, 1), round(a, 4)])

    p_bleed = 4.2 ** 2 / R_BLEED
    print(f"Bleed: {4.2 / R_BLEED * 1e3:.0f} mA at 4.2 V, {p_bleed:.2f} W per resistor, "
          f"about {p_bleed * R_TH_RESISTOR:.0f} K rise at {R_TH_RESISTOR:.0f} K/W")
    print(f"Estimate: 100 mAh / {4.1 / R_BLEED * 1e3:.0f} mA = {100 / (4.1 / R_BLEED * 1e3):.1f} h of bleeding needed\n")
    print(f"{'Case':15s} " + " ".join(f"cycle {k}" for k in range(1, 5)))
    for label, (log, eoc, ah) in res.items():
        print(f"{label:15s} spread at end of charge (mV): " + "  ".join(f"{s:6.1f}" for s in eoc))
        print(f"{'':15s} capacity delivered (Ah):     " + "  ".join(f"{a:6.3f}" for a in ah))
        bleed_h = log[:, 5:9].sum(0) * DT / 3600
        print(f"{'':15s} bleed time per cell (h): {np.round(bleed_h, 2)}")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True)
    for ax, (label, (log, eoc, _)) in zip(axes, res.items()):
        th = log[:, 0] / 3600
        spread = 1e3 * (log[:, 1:5].max(1) - log[:, 1:5].min(1))
        ax.plot(th, spread, color="#c43f3f" if "off" in label else "#1a9e5c", lw=0.8)
        ax.set_ylabel("Cell spread (mV)")
        ax.set_ylim(0, 160)
        ax.set_title(f"{label}: spread at end of charge, cycles 1-4: "
                     + ", ".join(f"{s:.0f}" for s in eoc) + " mV", fontsize=10)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("Time (h)")
    fig.suptitle("Passive top balancing, cell 3 starting 100 mAh low (60 ohm bleed)\n"
                 "Tall peaks = end of each 2 A discharge, where the weakest cell falls away first", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "plots" / "balancing.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote data/balancing.csv and plots/balancing.png under {ROOT}")


if __name__ == "__main__":
    main()
