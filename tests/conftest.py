"""Settings shared by the whole test suite."""
import os

# The learners' forests, CatBoost and networks run with fewer trees and seeds in the tests (the many walk-forward
# tests would otherwise take most of an hour). The pipeline itself never sets this; one test in test_ml.py checks
# the full settings.
os.environ.setdefault("SMR_ML_LIGHT", "1")
