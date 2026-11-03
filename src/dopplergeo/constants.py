"""Physical constants and defaults shared by every module."""

# Speed of sound in air at 20 degrees C, m/s. Real air at a given site is
# roughly 331.3 + 0.606 * T_celsius; a 10 degree error moves this by ~2%,
# which moves fitted aircraft speeds by the same ~2%.
SPEED_OF_SOUND_MPS: float = 343.0

# WGS84 mean earth radius, m. The local projections in geo.py assume a sphere,
# which is accurate to ~0.3% in distance over the <100 km scales used here.
EARTH_RADIUS_M: float = 6_371_008.8

FEET_TO_M: float = 0.3048
KNOTS_TO_MPS: float = 0.514444
FPM_TO_MPS: float = 0.3048 / 60.0

# Default candidate grid spacing (m) and refinement spacing (m).
COARSE_GRID_M: float = 200.0
FINE_GRID_M: float = 100.0

# Beyond this slant distance an aircraft is treated as inaudible.
MAX_HEARING_DISTANCE_M: float = 10_000.0
