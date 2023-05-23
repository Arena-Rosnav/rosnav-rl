import rospy
import math
import os

VALID_CONFIG_NAMES = ["hyperparameters.json", "training_config.yaml"]

REDUCTION_FACTOR = 3

RosnavEncoder = {
    "DefaultEncoder": {
        "lasers": rospy.get_param(
            os.path.join(rospy.get_namespace(), "laser/num_beams")
        ),
        "meta": 2 + 3,  # Goal + Vel,
        "lasers_to_adapted": lambda x: x,
    },
    "StackedEncoder": {
        "lasers": rospy.get_param(
            os.path.join(rospy.get_namespace(), "laser/num_beams")
        ),
        "meta": 2 + 3,  # Goal + Vel,
        "lasers_to_adapted": lambda x: x,
    },
    "ReducedEncoder": {
        "lasers": math.ceil(rospy.get_param("laser/num_beams") / REDUCTION_FACTOR),
        "meta": 2 + 3,  # Goal + Vel
        "lasers_to_adapted": lambda x: math.ceil(x / REDUCTION_FACTOR),
    },
    "UniformEncoder": {
        "lasers": 1200,
        "meta": 2 + 3 + 1 + 6,  # Goal + Vel + Radius + Max Vel
        "maxVelocity": {"x": [-5, 5], "y": [-5, 5], "angular": [-10, 10]},
    },
    "ReducedLaserEncoder": {
        "lasers": rospy.get_param(
            os.path.join(rospy.get_namespace(), "laser/reduced_num_laser_beams"),
            rospy.get_param(os.path.join(rospy.get_namespace(), "laser/num_beams")),
        ),
        "meta": 2 + 3,  # Goal + Vel
        "lasers_to_adapted": lambda _: rospy.get_param(
            os.path.join(rospy.get_namespace(), "laser/reduced_num_laser_beams")
        ),
    },
    "StackedReducedLaserEncoder": {
        "lasers": rospy.get_param(
            os.path.join(rospy.get_namespace(), "laser/reduced_num_laser_beams"),
            rospy.get_param(os.path.join(rospy.get_namespace(), "laser/num_beams")),
        ),
        "meta": 2 + 3,  # Goal + Vel,
        "lasers_to_adapted": lambda _: rospy.get_param(
            os.path.join(rospy.get_namespace(), "laser/reduced_num_laser_beams")
        ),
    },
}
