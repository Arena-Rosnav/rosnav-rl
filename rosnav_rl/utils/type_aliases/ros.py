from typing import TypeVar

import rospy

_RospyMessage = TypeVar("RospyMessage", bound=rospy.Message)
