import os
import inspect
# Get main directory
frame = inspect.currentframe()
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(
    inspect.getfile(frame)), '..'))
# MAIN_DIR = os.path.abspath(os.path.dirname(
#     inspect.getfile(frame)))

# Define figures directory
FIG_DIR = os.path.join(MAIN_DIR, 'figures')

# Define experimental data directory
DATA_DIR = os.path.join(MAIN_DIR, 'Li-data')

# Define experimental parameters directory
PARAM_DIR = os.path.join(MAIN_DIR, 'parameters')

del(os, inspect, frame)

ABS_TOL = 1e-7
REL_TOL = 1e-8
