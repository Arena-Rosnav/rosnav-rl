import argparse

import rospy

from rosnav.node import RosnavNode


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-ns", "--namespace", type=str, default=None)
    parser.add_argument("-a", "--agent", type=str, default=None)

    return parser.parse_known_args()[0]


if __name__ == "__main__":
    rospy.init_node("rosnav_node")
    args = parse_args()

    RosnavNode(namespace=args.namespace, agent=args.agent)

    while not rospy.is_shutdown():
        rospy.spin()
