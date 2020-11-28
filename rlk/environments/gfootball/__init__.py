import warnings

try:
    AVAILABLE = True
except ImportError:
    warnings.warn('gfootball is not installed.')
    AVAILABLE = False
