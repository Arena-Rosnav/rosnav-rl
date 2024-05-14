import argparse

import rospy
from dynamic_reconfigure.client import Client
from rosnav.node import RosnavNode


from rl_utils.utils.drl_switch.constants import (
    DMRC_SERVER,
    DMRC_SERVER_ACTION,
    MBF_COMPATIBLE_TYPE,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-ns", "--namespace", type=str, default=None)
    parser.add_argument("-a", "--agent", type=str, default=None)

    return parser.parse_known_args()[0]


if __name__ == "__main__":
    rospy.init_node("rosnav_node")
    args = parse_args()

    node = RosnavNode(namespace=args.namespace, agent=args.agent)

    dmrc_mblegacy = Client(
        node.ns(DMRC_SERVER)(MBF_COMPATIBLE_TYPE.LOCAL.rosnav.value),
    )
    dmrc_mbf = Client(
        node.ns(DMRC_SERVER_ACTION),
    )

    # Set the active agent in MoveBaseFlex and MoveBaseLegacyRelay
    dmrc_mbf.update_configuration(
        {"base_local_planner": MBF_COMPATIBLE_TYPE.LOCAL.rosnav.value}
    )
    dmrc_mblegacy.update_configuration({"active_agent": node.agent_name})

    while not rospy.is_shutdown():
        rospy.spin()
