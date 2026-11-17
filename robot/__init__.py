"""Habitat-free real-robot (Jackal) navigation wrapper.

Runs the sim's objnav agent on the real Jackal with NO habitat import, by REUSING the
decision code (RobotAgent subclasses LastMileFMMAgent) fed by odom / RGB / depth over
one SSH channel (robot/jackal.py). A matched-input regression check verifies shared-code
identity; default sim and deployed-robot inputs/configuration are intentionally different.
See robot/README.md.
"""
from robot.config import RobotConfig
from robot.robot_env import RobotEnv
from robot.agent import RobotAgent
from robot.jackal import JackalConnection, JackalConnectionConfig

__all__ = ["RobotConfig", "RobotEnv", "RobotAgent",
           "JackalConnection", "JackalConnectionConfig"]
