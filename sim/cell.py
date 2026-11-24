"""4S Li-ion pack model used as the 'true' plant in every simulation.

Each cell: OCV(SOC) + R0(SOC) + two RC pairs (fast and slow) + lumped thermal model.
The estimators deliberately use a simpler 1-RC model, so the simulation shows
real model mismatch, as on hardware.

Sign convention: positive current = discharge.
The OCV curve is a representative NMC 18650 curve. Replace it with your Phase 3 data.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

SOC_PTS = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
OCV_PTS = np.array([3.00, 3.35, 3.46, 3.57, 3.63, 3.68, 3.73, 3.80, 3.88, 3.97, 4.07, 4.19])


def ocv(soc):
    return np.interp(np.clip(soc, 0.0, 1.0), SOC_PTS, OCV_PTS)


def docv(soc):
    """Slope dOCV/dSOC (V per unit SOC) of the piecewise-linear curve."""
    s = np.clip(soc, 0.0, 0.999999)
    i = np.clip(np.searchsorted(SOC_PTS, s, side="right") - 1, 0, len(SOC_PTS) - 2)
    return (OCV_PTS[i + 1] - OCV_PTS[i]) / (SOC_PTS[i + 1] - SOC_PTS[i])


def soc_from_ocv(v):
    return np.interp(v, OCV_PTS, SOC_PTS)


@dataclass(frozen=True)
class CellParams:
    q_ah: float = 2.5          # capacity
    r0: float = 0.035          # ohmic resistance at high SOC
    r0_low_soc: float = 0.6    # R0 rises by this fraction towards SOC 0
    r1: float = 0.015          # fast RC (charge transfer)
    tau1: float = 30.0
    r2: float = 0.008          # slow RC (diffusion) - not in the 1-RC estimator model
    tau2: float = 400.0
    eta_chg: float = 0.995     # coulombic efficiency when charging
    c_th: float = 45.0         # J/K, about 45 g x 1 J/(g K)
    r_th: float = 15.0         # K/W to ambient


def r0_of(p: CellParams, soc):
    return p.r0 * (1 + p.r0_low_soc * (1 - np.clip(soc, 0, 1)) ** 3)


def make_cells(n: int = 4, rng=None, q_spread=0.02, r_spread=0.08, base: CellParams = CellParams()):
    """Cells from one batch: capacity and resistance vary by a few percent."""
    rng = np.random.default_rng(42) if rng is None else rng
    return [replace(base, q_ah=base.q_ah * (1 + q_spread * rng.standard_normal()),
                    r0=base.r0 * (1 + r_spread * rng.standard_normal())) for _ in range(n)]


@dataclass
class Pack:
    cells: list
    soc: np.ndarray
    t_amb: float = 25.0
    v1: np.ndarray = field(default=None)
    v2: np.ndarray = field(default=None)
    temp: np.ndarray = field(default=None)

    def __post_init__(self):
        n = len(self.cells)
        self.soc = np.array(self.soc, dtype=float)
        self.v1 = np.zeros(n) if self.v1 is None else self.v1
        self.v2 = np.zeros(n) if self.v2 is None else self.v2
        self.temp = np.full(n, self.t_amb) if self.temp is None else self.temp
        self._q = np.array([c.q_ah for c in self.cells])
        self._r1 = np.array([c.r1 for c in self.cells])
        self._t1 = np.array([c.tau1 for c in self.cells])
        self._r2 = np.array([c.r2 for c in self.cells])
        self._t2 = np.array([c.tau2 for c in self.cells])
        self._eta = np.array([c.eta_chg for c in self.cells])
        self._cth = np.array([c.c_th for c in self.cells])
        self._rth = np.array([c.r_th for c in self.cells])

    @property
    def r0(self):
        return np.array([r0_of(c, s) for c, s in zip(self.cells, self.soc)])

    def step(self, i_pack: float, dt: float, i_extra=None):
        """Advance by dt. i_extra = per-cell extra discharge current (bleed, divider drain)."""
        i_cell = i_pack + (0.0 if i_extra is None else np.asarray(i_extra))
        i_cell = np.broadcast_to(i_cell, self.soc.shape).astype(float)
        eta = np.where(i_cell < 0, self._eta, 1.0)
        r0 = self.r0
        heat = i_cell ** 2 * r0 + i_cell * (self.v1 + self.v2)
        self.soc = self.soc - eta * i_cell * dt / (3600 * self._q)
        a1, a2 = np.exp(-dt / self._t1), np.exp(-dt / self._t2)
        self.v1 = a1 * self.v1 + self._r1 * (1 - a1) * i_cell
        self.v2 = a2 * self.v2 + self._r2 * (1 - a2) * i_cell
        self.temp = self.temp + dt * (heat - (self.temp - self.t_amb) / self._rth) / self._cth
        return self.terminal(i_cell)

    def terminal(self, i_cell):
        i_cell = np.broadcast_to(i_cell, self.soc.shape)
        return ocv(self.soc) - i_cell * self.r0 - self.v1 - self.v2
