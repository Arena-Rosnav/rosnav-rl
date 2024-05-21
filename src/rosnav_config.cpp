#include <rosnav_config.h>


void rosnav::RosnavConfig::loadRosParamFromNodeHandle(const ros::NodeHandle& nh)
{
    nh.param("max_vel_x", robot.max_vel_x, robot.max_vel_x);
    nh.param("max_vel_x_backwards", robot.max_vel_x_backwards, robot.max_vel_x_backwards);
    nh.param("max_vel_y", robot.max_vel_y, robot.max_vel_y);
    nh.param("max_vel_trans", robot.max_vel_trans, robot.max_vel_trans);
    nh.param("max_vel_theta", robot.max_vel_theta, robot.max_vel_theta);
    nh.param("use_proportional_saturation", robot.use_proportional_saturation, robot.use_proportional_saturation);
}

void rosnav::RosnavConfig::reconfigure(RosnavLocalPlannerReconfigureConfig& cfg)
{
    robot.max_vel_x = cfg.max_vel_x;
    robot.max_vel_x_backwards = cfg.max_vel_x_backwards;
    robot.max_vel_y = cfg.max_vel_y;
    robot.max_vel_trans = robot.max_vel_trans;
    robot.max_vel_theta = cfg.max_vel_theta;
    robot.use_proportional_saturation = cfg.use_proportional_saturation;
}

void rosnav::RosnavConfig::checkParameters() const
{
    // if (robot.max_vel_x < 0.0)
    // {
    //     ROS_WARN("RosnavLocalPlanner() Param Warning: max_vel_x should be positive. Resetting to default value.");
    //     robot.max_vel_x = 2.0;
    // }
    // if (robot.max_vel_x_backwards < 0.0)
    // {
    //     ROS_WARN("RosnavLocalPlanner() Param Warning: max_vel_x_backwards should be positive. Resetting to default value.");
    //     robot.max_vel_x_backwards = 2.0;
    // }
    // if (robot.max_vel_y < 0.0)
    // {
    //     ROS_WARN("RosnavLocalPlanner() Param Warning: max_vel_y should be positive. Resetting to default value.");
    //     robot.max_vel_y = 0.0;
    // }
}