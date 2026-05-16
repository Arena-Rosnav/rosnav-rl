"""Custom policy module — MLP_ARENA2D (legacy ROS1 policy removed).

The original custom policy implementation depended on ROS1 (rospy/rospkg) for
observation-space resolution.  Policies are now defined via SB3's built-in
policy classes and the observation manager; see base_policy.py.
"""
