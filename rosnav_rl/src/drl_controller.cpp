#include "rosnav_rl/drl_controller.hpp"

namespace rosnav_rl_planner
{

void DRLController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & /*parent*/,
  std::string /*name*/,
  std::shared_ptr<tf2_ros::Buffer> /*tf*/,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> /*costmap_ros*/)
{
  // Initialization, load your DRL model, etc.
}

void DRLController::cleanup()
{
  // Cleanup resources
}

void DRLController::activate()
{
  // Activate resources
}

void DRLController::deactivate()
{
  // Deactivate resources
}

geometry_msgs::msg::TwistStamped DRLController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & /*pose*/,
  const geometry_msgs::msg::Twist & /*velocity*/,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  geometry_msgs::msg::TwistStamped cmd_vel;
  // Minimal: return zero velocity
  cmd_vel.twist.linear.x = 0.0;
  cmd_vel.twist.angular.z = 0.0;

  // debug logging
  RCLCPP_INFO(rclcpp::get_logger("DRLController"), "computeVelocityCommands called, returning zero velocity");
  return cmd_vel;
}

void DRLController::setPlan(const nav_msgs::msg::Path & /*path*/)
{
  // Store or process the global plan if needed
}

void DRLController::setSpeedLimit(const double & /*speed_limit*/, const bool & /*percentage*/)
{
  // Optionally implement speed limiting
}

}  // namespace rosnav_rl

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(rosnav_rl_planner::DRLController, nav2_core::Controller)
