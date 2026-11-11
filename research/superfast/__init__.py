"""
Two-scale (frozen-Sun local diurnal + seasonal) Yarkovsky method for fast and super-fast tumbling asteroids.

Extends the IAC 2026 frozen-Sun method (~/Desktop/Tempestovsky-main/src/model/reduced_diurnal.py) to general
triaxial torque-free rotation (SAM and LAM), eccentric Keplerian orbits and the seasonal contribution, using the
full solver's own conventions, initial state, column kernel and Yarkovsky formula, so that any campaign manifest
line can be computed. See research/superfast/METHOD.md.
"""
