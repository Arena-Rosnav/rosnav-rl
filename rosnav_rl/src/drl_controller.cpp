#include "rosnav_rl/drl_controller.hpp"
#include <thread>

namespace rosnav_rl_planner
{

void DRLController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> /*tf*/,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> /*costmap_ros*/)
{
  // Initialization, load your DRL model, etc.
  node_ = parent;
  auto node = node_.lock();
  plugin_name_ = name;

  logger_ = node->get_logger();
  clock_ = node->get_clock();

  // Create a MutuallyExclusive callback group for service calls
  // This ensures service calls don't interfere with other controller operations
  callback_group_ = node->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);

  // Get service name parameter, default to "get_command" 
  std::string service_name = "get_command";
  node->declare_parameter(plugin_name_ + ".service_name", service_name);
  node->get_parameter(plugin_name_ + ".service_name", service_name);
  
  RCLCPP_INFO(logger_, "Creating get_command client for service: %s", service_name.c_str());
  
  // Create client with the callback group for proper execution
  client_ = node->create_client<rosnav_rl_msgs::srv::GetCommand>(
    service_name, 
    rmw_qos_profile_services_default,
    callback_group_);

  // Create publisher for global plan
  global_pub_ = node->create_publisher<nav_msgs::msg::Path>("global_plan", 1);
}

void DRLController::cleanup()
{
  // Cleanup resources
  RCLCPP_INFO(
    logger_,
    "[ROSNAV_CONTROLLER] Cleaning up controller: %s",
    plugin_name_.c_str()
  );
  global_pub_.reset();
}

void DRLController::activate()
{
  // Activate resources
  RCLCPP_INFO(
    logger_,
    "[ROSNAV_CONTROLLER] Activating controller: %s",
    plugin_name_.c_str()
  );
  global_pub_->on_activate();
}

void DRLController::deactivate()
{
  // Deactivate resources
  RCLCPP_INFO(
    logger_,
    "[ROSNAV_CONTROLLER] Deactivating controller: %s",
    plugin_name_.c_str()
  );
  global_pub_->on_deactivate();
}

geometry_msgs::msg::TwistStamped DRLController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{
  (void)goal_checker;

  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header.frame_id = pose.header.frame_id;
  cmd_vel.header.stamp = clock_->now();
  
  RCLCPP_DEBUG(logger_, "[ROSNAV_CONTROLLER] computeVelocityCommands called - checking service availability");
  
  if (!client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(logger_, "[ROSNAV_CONTROLLER] Service not available, stopping robot");
    cmd_vel.twist.linear.x = 0.0;
    cmd_vel.twist.angular.z = 0.0;
    return cmd_vel;
  }

  RCLCPP_DEBUG(logger_, "[ROSNAV_CONTROLLER] Service available, sending request");
  auto request = std::make_shared<rosnav_rl_msgs::srv::GetCommand::Request>();
  auto future = client_->async_send_request(request);

  // Simple polling loop without any ROS executor interactions
  auto node = node_.lock();
  auto start = node->now();
  while (rclcpp::ok()) {
    if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      cmd_vel.twist = future.get()->twist;
      RCLCPP_DEBUG(
        logger_,
        "[ROSNAV_CONTROLLER] Received velocity command: linear_x=%.2f, linear_y=%.2f, angular_z=%.2f",
        cmd_vel.twist.linear.x, cmd_vel.twist.linear.y, cmd_vel.twist.angular.z);
      break;
    }
    if ((node->now() - start).seconds() > 2.0) {  // Timeout less than environment timeout (10s)
      RCLCPP_ERROR(logger_, "[ROSNAV_CONTROLLER] Timeout waiting for get_command service, stopping robot");
      cmd_vel.twist.linear.x = 0.0;
      cmd_vel.twist.angular.z = 0.0;
      break;
    }
    // Simple sleep without ROS rate/executor involvement
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
    return cmd_vel;
  }

void DRLController::setPlan(const nav_msgs::msg::Path & path)
{
  // Store or process the global plan if needed
  global_plan_ = path;
  global_pub_->publish(path);
}

void DRLController::setSpeedLimit(const double & /*speed_limit*/, const bool & /*percentage*/)
{
  // Optionally implement speed limiting
}

}  // namespace rosnav_rl

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(rosnav_rl_planner::DRLController, nav2_core::Controller)
