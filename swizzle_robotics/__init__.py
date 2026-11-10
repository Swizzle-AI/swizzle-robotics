"""
Swizzle Robotics: 80.5 Hz Visual Servoing, Quintic Spline Smoothing, and Latent World Models.
"""
from .servoing import VisualServoingLoop, ServoingConfig
from .world_model import LatentWorldModel, MPCPlanner
from .spline import QuinticSplineTrajectory

__version__ = "0.1.0"
__all__ = [
    "VisualServoingLoop",
    "ServoingConfig",
    "LatentWorldModel",
    "MPCPlanner",
    "QuinticSplineTrajectory",
]
