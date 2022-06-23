import rospy
import rospkg
import yaml
import os


def get_robot_yaml_path():
    robot_model = rospy.get_param("model")

    simulation_setup_path = rospkg.RosPack().get_path("arena-simulation-setup")
    return os.path.join(
        simulation_setup_path, "robot", robot_model, f"{robot_model}.model.yaml"
    )


def get_laser_from_robot_yaml():
    robot_yaml_path = get_robot_yaml_path()

    with open(robot_yaml_path, "r") as fd:
        robot_data = yaml.safe_load(fd)

        for plugin in robot_data["plugins"]:
            if plugin["type"] == "Laser":
                laser_angle_min = plugin["angle"]["min"]
                laser_angle_max = plugin["angle"]["max"]
                laser_angle_increment = plugin["angle"]["increment"]
                
                _L = int(
                    round(
                        (laser_angle_max - laser_angle_min) / laser_angle_increment
                    )
                )
                
                return _L, laser_angle_min, laser_angle_max, laser_angle_increment


def get_robot_state_size():
    if not rospy.get_param("actions_in_obs", default=False):
        return 2  # robot state size

    return 2 + 3  # rho, theta, linear x, linear y, angular z
