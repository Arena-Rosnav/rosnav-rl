#include "rosnav_rl/drl_controller.hpp"
#include <thread>
#include <cmath>

namespace rosnav_rl_planner
{

void DRLController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  // Initialization, load your DRL model, etc.
  node_ = parent;
  auto node = node_.lock();
  plugin_name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;

  logger_ = node->get_logger();
  clock_ = node->get_clock();

  // Create a MutuallyExclusive callback group for service calls
  // This ensures service calls don't interfere with other controller operations
  callback_group_ = node->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);

  // Get service name parameter, default to "get_command" 
  std::string service_name = "get_command";
  rcl_interfaces::msg::ParameterDescriptor service_name_descriptor;
  service_name_descriptor.description = "The name of the service to call to get the velocity commands from the DRL agent.";
  node->declare_parameter(plugin_name_ + ".service_name", service_name, service_name_descriptor, true);
  node->get_parameter(plugin_name_ + ".service_name", service_name);
  
  double subgoal_frequency;
  rcl_interfaces::msg::ParameterDescriptor subgoal_frequency_descriptor;
  subgoal_frequency_descriptor.description = "The frequency in Hz at which the subgoal is published.";
  node->declare_parameter(plugin_name_ + ".subgoal_frequency", 2.0, subgoal_frequency_descriptor, true);
  node->get_parameter(plugin_name_ + ".subgoal_frequency", subgoal_frequency);

  rcl_interfaces::msg::ParameterDescriptor min_lookahead_dist_descriptor;
  min_lookahead_dist_descriptor.description = "The minimum lookahead distance in meters. This is the shortest distance the robot will look ahead on the path, used at low speeds.";
  node->declare_parameter(plugin_name_ + ".min_lookahead_dist", 0.5, min_lookahead_dist_descriptor, true);
  node->get_parameter(plugin_name_ + ".min_lookahead_dist", min_lookahead_dist_);

  rcl_interfaces::msg::ParameterDescriptor max_lookahead_dist_descriptor;
  max_lookahead_dist_descriptor.description = "The maximum lookahead distance in meters. This is the longest distance the robot will look ahead on the path, used at high speeds.";
  node->declare_parameter(plugin_name_ + ".max_lookahead_dist", 2.5, max_lookahead_dist_descriptor, true);
  node->get_parameter(plugin_name_ + ".max_lookahead_dist", max_lookahead_dist_);

  rcl_interfaces::msg::ParameterDescriptor lookahead_time_descriptor;
  lookahead_time_descriptor.description = "The time in seconds to look ahead on the path. The actual lookahead distance is calculated as `current_velocity * lookahead_time` and clamped between `min_lookahead_dist` and `max_lookahead_dist`.";
  node->declare_parameter(plugin_name_ + ".lookahead_time", 1.5, lookahead_time_descriptor, true);
  node->get_parameter(plugin_name_ + ".lookahead_time", lookahead_time_);

  RCLCPP_INFO(logger_, "Creating get_command client for service: %s", service_name.c_str());
  
  // Create client with the callback group for proper execution
  client_ = node->create_client<rosnav_rl_msgs::srv::GetCommand>(
    service_name, 
    rmw_qos_profile_services_default,
    callback_group_);

  // Create publisher for global plan
  global_pub_ = node->create_publisher<nav_msgs::msg::Path>("global_plan", 1);
  subgoal_pub_ = node->create_publisher<geometry_msgs::msg::PoseStamped>("subgoal", 1);

  // Create a timer for periodically publishing the subgoal
  subgoal_timer_ = node->create_wall_timer(
    std::chrono::duration<double>(1.0 / subgoal_frequency),
    std::bind(&DRLController::publishSubgoal, this),
    callback_group_);
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
  subgoal_pub_.reset();
  subgoal_timer_.reset();
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
  subgoal_pub_->on_activate();
  subgoal_timer_->reset(); // Resets the timer to start counting
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
  subgoal_pub_->on_deactivate();
  subgoal_timer_->cancel();
}

geometry_msgs::msg::TwistStamped DRLController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{
  (void)goal_checker;
  current_velocity_ = velocity;

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
      RCLCPP_INFO(
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

void DRLController::publishSubgoal()
{
  RCLCPP_DEBUG(logger_, "[ROSNAV_CONTROLLER] Attempting to publish subgoal using adaptive lookahead.");
  if (global_plan_.poses.empty() || !costmap_ros_) {
    RCLCPP_DEBUG(logger_, "[ROSNAV_CONTROLLER] Global plan is empty or costmap_ros is null. Cannot publish subgoal.");
    return;
  }

  // Get robot's current pose
  geometry_msgs::msg::PoseStamped robot_pose;
  if (!costmap_ros_->getRobotPose(robot_pose)) {
    RCLCPP_ERROR(logger_, "[ROSNAV_CONTROLLER] Failed to get robot pose. Cannot publish subgoal.");
    return;
  }

  // Adaptive lookahead distance
  double lookahead_dist = current_velocity_.linear.x * lookahead_time_;
  lookahead_dist = std::clamp(lookahead_dist, min_lookahead_dist_, max_lookahead_dist_);
  RCLCPP_DEBUG(logger_, "[ROSNAV_CONTROLLER] Adaptive lookahead distance: %.2f", lookahead_dist);

  // Find the point on the global plan that is 'lookahead_dist' away from the robot
  for (const auto & plan_pose : global_plan_.poses) {
    double dist = std::hypot(
      plan_pose.pose.position.x - robot_pose.pose.position.x,
      plan_pose.pose.position.y - robot_pose.pose.position.y);

    if (dist >= lookahead_dist) {
      geometry_msgs::msg::PoseStamped subgoal = plan_pose;
      subgoal.header.stamp = clock_->now();
      subgoal_pub_->publish(subgoal);
      RCLCPP_DEBUG(
        logger_, "[ROSNAV_CONTROLLER] Published subgoal (lookahead): x=%.2f, y=%.2f",
        subgoal.pose.position.x, subgoal.pose.position.y);
      return;
    }
  }

  // If no point is far enough, use the last point of the plan (the goal)
  if (!global_plan_.poses.empty()) {
    geometry_msgs::msg::PoseStamped subgoal = global_plan_.poses.back();
    subgoal.header.stamp = clock_->now();
    subgoal_pub_->publish(subgoal);
    RCLCPP_DEBUG(
      logger_, "[ROSNAV_CONTROLLER] Published subgoal (end of plan): x=%.2f, y=%.2f",
      subgoal.pose.position.x, subgoal.pose.position.y);
    return;
  }

  RCLCPP_DEBUG(logger_, "[ROSNAV_CONTROLLER] No valid subgoal found.");
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
