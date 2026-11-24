"""Protection state machine: the firmware twin.

Written so it can be ported line by line to C on the ESP32: one update() per 10 Hz tick,
no allocation, plain numbers in and out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INIT, IDLE, CHARGING, DISCHARGING, FAULT = "INIT", "IDLE", "CHARGING", "DISCHARGING", "FAULT"
FAULTS = ("OV", "UV", "OCC", "OCD", "OTC", "OTD", "UTC", "SENSOR")


@dataclass(frozen=True)
class Limits:
    ov_trip: float = 4.20
    ov_clear: float = 4.10
    ov_delay: float = 0.5
    uv_trip: float = 3.00
    uv_clear: float = 3.20
    uv_delay: float = 0.5
    occ_trip: float = 2.5      # A, charge
    occ_delay: float = 0.2
    ocd_trip: float = 5.0      # A, discharge
    ocd_delay: float = 0.1
    otc_trip: float = 45.0     # degC, while charging
    otd_trip: float = 60.0     # degC, any time
    ut_c_trip: float = 0.0     # degC, while charging
    t_hyst: float = 5.0
    t_delay: float = 1.0
    i_idle: float = 0.05       # A, below this the pack is idle
    v_valid: tuple = (0.5, 5.0)
    t_valid: tuple = (-30.0, 100.0)
    stale_s: float = 0.5
    sensor_delay: float = 0.2


class Protection:
    def __init__(self, limits: Limits = Limits(), dt: float = 0.1, latch: bool = True,
                 use_debounce: bool = True, use_hysteresis: bool = True):
        self.l, self.dt, self.latch = limits, dt, latch
        self.use_debounce, self.use_hysteresis = use_debounce, use_hysteresis
        self.timers = dict.fromkeys(FAULTS, -1.0)
        self.active = set()
        self.state = INIT

    def _conditions(self, v, i, t, age):
        l = self.l
        charging = i < -l.i_idle
        bad = (np.any(v < l.v_valid[0]) or np.any(v > l.v_valid[1]) or np.any(t < l.t_valid[0])
               or np.any(t > l.t_valid[1]) or age > l.stale_s or np.any(~np.isfinite(v)))
        return {"OV": v.max() > l.ov_trip, "UV": v.min() < l.uv_trip,
                "OCC": -i > l.occ_trip, "OCD": i > l.ocd_trip,
                "OTC": charging and t.max() > l.otc_trip, "OTD": t.max() > l.otd_trip,
                "UTC": charging and t.min() < l.ut_c_trip, "SENSOR": bool(bad)}

    def _clear(self, name, v, i, t):
        l = self.l
        if not self.use_hysteresis:
            return True
        return {"OV": v.max() < l.ov_clear, "UV": v.min() > l.uv_clear,
                "OCC": -i < 0.8 * l.occ_trip, "OCD": i < 0.8 * l.ocd_trip,
                "OTC": t.max() < l.otc_trip - l.t_hyst, "OTD": t.max() < l.otd_trip - l.t_hyst,
                "UTC": t.min() > l.ut_c_trip + l.t_hyst, "SENSOR": True}[name]

    def _delay(self, name):
        if not self.use_debounce:
            return 0.0
        l = self.l
        return {"OV": l.ov_delay, "UV": l.uv_delay, "OCC": l.occ_delay, "OCD": l.ocd_delay,
                "OTC": l.t_delay, "OTD": l.t_delay, "UTC": l.t_delay, "SENSOR": l.sensor_delay}[name]

    def update(self, v, i, t, age=0.0):
        v, t = np.asarray(v, float), np.asarray(t, float)
        cond = self._conditions(v, i, t, age)
        for name, c in cond.items():
            # Timer = time since the first sample that saw the condition; trips once it reaches the delay.
            if c:
                self.timers[name] = self.timers[name] + self.dt if self.timers[name] >= 0 else 0.0
            else:
                self.timers[name] = -1.0
            if c and self.timers[name] >= self._delay(name) - 1e-9:
                self.active.add(name)
            elif not self.latch and name in self.active and not c and self._clear(name, v, i, t):
                self.active.discard(name)
        if self.active:
            self.state = FAULT
        elif i > self.l.i_idle:
            self.state = DISCHARGING
        elif i < -self.l.i_idle:
            self.state = CHARGING
        else:
            self.state = IDLE
        return self.state

    @property
    def fets_on(self):
        return self.state != FAULT

    def reset(self, v, i, t, age=0.0):
        """Deliberate reset. Refused while any latched fault is still outside its clear level."""
        v, t = np.asarray(v, float), np.asarray(t, float)
        cond = self._conditions(v, i, t, age)
        if any(cond[n] or not self._clear(n, v, i, t) for n in self.active):
            return False
        self.active.clear()
        self.timers = dict.fromkeys(FAULTS, -1.0)
        self.state = IDLE
        return True


def hw_short_circuit(i_sc=100.0, tau=20e-6, threshold=20.0, blanking=100e-6, turn_off=5e-6, dt=1e-6, t_end=2e-3):
    """Comparator on the shunt, with a blanking delay, driving the MOSFET gate directly.
    Returns trip time after short onset and the I^2 t let through."""
    t = np.arange(0, t_end, dt)
    i = i_sc * (1 - np.exp(-t / tau))
    above = np.where(i > threshold)[0]
    t_off = t[above[0]] + blanking + turn_off
    i_lt = np.where(t < t_off, i, 0.0)
    return t_off, float(np.sum(i_lt ** 2) * dt)
