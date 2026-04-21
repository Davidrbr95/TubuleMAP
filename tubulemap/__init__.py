
import logging
from .widgets import PointsWidget
from .utils.misc_utils import normal_round
__version__ = "0.1.0"

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# If you want to expose certain functions or classes to be directly accessible from the package:
__all__ = [
    "PointsWidget",
    "normal_round",
]
