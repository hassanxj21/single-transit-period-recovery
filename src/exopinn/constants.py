"""Physical constants in the unit system used throughout this project.

Units: distance in AU, mass in solar masses, time in days.
In this system GM_sun = 4*pi^2 / 365.25^2 = 2.959e-4 AU^3 / (M_sun day^2).
"""

import numpy as np

G = 2.959122083e-4        # AU^3 / (M_sun * day^2)
R_SUN_AU = 4.6491e-3      # 1 solar radius in AU
R_JUP_R_SUN = 0.10045     # 1 Jupiter radius in solar radii
FOURPI2 = 4.0 * np.pi**2

# Habitable zone bracket used for flagging candidates (equilibrium temperature, K)
HZ_TEMP_MIN = 200.0
HZ_TEMP_MAX = 320.0
BOND_ALBEDO = 0.3

# TESS
SECTOR_BASELINE_DAYS = 27.0
