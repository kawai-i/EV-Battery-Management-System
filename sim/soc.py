"""SOC estimators: the firmware twins. Plain floats so each one ports directly to C."""
from __future__ import annotations

import math

import numpy as np

from .cell import docv, ocv, soc_from_ocv


class CoulombCounter:
    name = "Coulomb counting"

    def __init__(self, soc0, q_ah, eta_chg=0.995):
        self.soc, self.q, self.eta = float(soc0), float(q_ah), eta_chg

    def update(self, i, v, dt):
        eta = self.eta if i < 0 else 1.0
        self.soc -= eta * i * dt / (3600 * self.q)
        return self.soc


class OcvAnchoredCC(CoulombCounter):
    """Coulomb counting, re-anchored to the OCV table after each long enough rest.
    The correction is weighted by the OCV slope: where the curve is flat, a few mV of
    voltage error is many % of SOC, so the anchor is trusted less."""
    name = "OCV + Coulomb counting"

    def __init__(self, soc0, q_ah, rest_s=600.0, i_rest=0.05, **kw):
        super().__init__(soc0, q_ah, **kw)
        self.rest_s, self.i_rest, self.rest_t, self.anchored = rest_s, i_rest, 0.0, False

    def update(self, i, v, dt):
        super().update(i, v, dt)
        if abs(i) < self.i_rest:
            self.rest_t += dt
            if self.rest_t >= self.rest_s and not self.anchored:
                s_ocv = float(soc_from_ocv(v))
                w = min(1.0, max(0.3, float(docv(s_ocv)) / 1.2))
                self.soc += w * (s_ocv - self.soc)
                self.anchored = True
        else:
            self.rest_t, self.anchored = 0.0, False
        return self.soc


class EKF:
    """Extended Kalman filter on [SOC, V_RC] with the fitted 1-RC model and an R0(SOC) lookup."""
    name = "EKF (1-RC)"

    def __init__(self, soc0, q_ah, ecm, p0_soc=0.1, q_soc=1e-8, q_v=2e-7, r_v=(8e-3) ** 2, eta_chg=0.995):
        self.x0, self.x1 = float(soc0), 0.0
        self.p00, self.p01, self.p11 = p0_soc ** 2, 0.0, (5e-3) ** 2
        self.q, self.eta = float(q_ah), eta_chg
        self.ecm, self.qs, self.qv, self.r = ecm, q_soc, q_v, r_v
        self.a = None

    def _r0(self, soc):
        return float(np.interp(soc, self.ecm["soc"], self.ecm["r0"]))

    @property
    def soc(self):
        return self.x0

    def update(self, i, v, dt):
        if self.a is None:
            self.a = math.exp(-dt / self.ecm["tau"])
        a, r1 = self.a, self.ecm["r1"]
        eta = self.eta if i < 0 else 1.0
        # predict
        self.x0 -= eta * i * dt / (3600 * self.q)
        self.x1 = a * self.x1 + r1 * (1 - a) * i
        p00 = self.p00 + self.qs
        p01 = a * self.p01
        p11 = a * a * self.p11 + self.qv
        # update with the measured cell voltage
        s = min(max(self.x0, 0.0), 1.0)
        h0, h1 = float(docv(s)), -1.0
        y = v - (float(ocv(s)) - i * self._r0(s) - self.x1)
        ph0 = p00 * h0 + p01 * h1
        ph1 = p01 * h0 + p11 * h1
        S = h0 * ph0 + h1 * ph1 + self.r
        k0, k1 = ph0 / S, ph1 / S
        self.x0 += k0 * y
        self.x1 += k1 * y
        self.p00 = p00 - k0 * ph0
        self.p01 = p01 - k0 * ph1
        self.p11 = p11 - k1 * ph1
        return self.x0
