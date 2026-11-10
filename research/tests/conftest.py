import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
REPO = RESEARCH.parent
for p in (str(RESEARCH), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Test states: inertia of a b/a = 0.8, c/a = 0.6 ellipsoid (I1 = 0.6098, I2 = 0.8293, separatrix 1/I2 = 1.2059)
I_ELL = (0.6097560975609756, 0.8292682926829268)
SAM_E, LAM_E = 1.10, 1.45
