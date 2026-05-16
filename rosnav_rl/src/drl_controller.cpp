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

  RCLCPP_INFO(logger_, "[ROSNAV_CONTROLLER] Configuring plugin '%s' — connecting to service '%s'", plugin_name_.c_str(), service_name.c_str());
  
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
  
  RCLCPP_DEBUG(logger_, "[ROSNAV_CONTROLLER] computeVelocityCommands called — checking service availability");

  if (!client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR_THROTTLE(logger_, *clock_, 5000,
      "[ROSNAV_CONTROLLER] Service '%s' not available after 1 s - stopping robot. ",
      client_->get_service_name());
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
      auto result = future.get();
      // response.action is [linear_x, linear_y, angular_z] for mobile robots
      if (result->action.size() >= 3) {
        cmd_vel.twist.linear.x  = result->action[0];
        cmd_vel.twist.linear.y  = result->action[1];
        cmd_vel.twist.angular.z = result->action[2];
      } else if (result->action.size() == 2) {
        // differential_drive may omit linear_y
        cmd_vel.twist.linear.x  = result->action[0];
        cmd_vel.twist.angular.z = result->action[1];
      }
      RCLCPP_DEBUG(
        logger_,
        "[ROSNAV_CONTROLLER] Received command: linear_x=%.2f, linear_y=%.2f, angular_z=%.2f",
        cmd_vel.twist.linear.x, cmd_vel.twist.linear.y, cmd_vel.twist.angular.z);
      break;
    }
    if ((node->now() - start).seconds() > 2.0) {  // Timeout less than environment timeout (10s)
      RCLCPP_ERROR(logger_, "[ROSNAV_CONTROLLER] Timeout waiting for '%s' response (>2 s) — stopping robot",
        client_->get_service_name());
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
  if (global_plan_.poses.empty() || !costmap_ros_) {
    RCLCPP_DEBUG(logger_, "[ROSNAV_CONTROLLER] No global plan yet — skipping subgoal publish");
    return;
  }

  // Get robot's current pose
  geometry_msgs::msg::PoseStamped robot_pose;
  if (!costmap_ros_->getRobotPose(robot_pose)) {
    RCLCPP_ERROR_THROTTLE(logger_, *clock_, 5000,
      "[ROSNAV_CONTROLLER] Failed to get robot pose — TF transform from '%s' to '%s' not available",
      costmap_ros_->getGlobalFrameID().c_str(), costmap_ros_->getBaseFrameID().c_str());
    return;
  }

  // Adaptive lookahead distance
  double lookahead_dist = current_velocity_.linear.x * lookahead_time_;
  lookahead_dist = std::clamp(lookahead_dist, min_lookahead_dist_, max_lookahead_dist_);

  // Step 1: Find the closest point on the path to the robot
  size_t closest_idx = 0;
  double min_dist = std::numeric_limits<double>::max();
  
  for (size_t i = 0; i < global_plan_.poses.size(); ++i) {
    double dist = std::hypot(
      global_plan_.poses[i].pose.position.x - robot_pose.pose.position.x,
      global_plan_.poses[i].pose.position.y - robot_pose.pose.position.y);
    
    if (dist < min_dist) {
      min_dist = dist;
      closest_idx = i;
    }
  }

  // Step 2: Traverse forward along the path, accumulating arc length
  double accumulated_dist = 0.0;
  size_t subgoal_idx = closest_idx;
  
  for (size_t i = closest_idx; i < global_plan_.poses.size() - 1; ++i) {
    double segment_length = std::hypot(
      global_plan_.poses[i + 1].pose.position.x - global_plan_.poses[i].pose.position.x,
      global_plan_.poses[i + 1].pose.position.y - global_plan_.poses[i].pose.position.y);
    
    accumulated_dist += segment_length;
    
    if (accumulated_dist >= lookahead_dist) {
      subgoal_idx = i + 1;
      break;
    }
    subgoal_idx = i + 1;
  }

  // Step 3: Publish the selected subgoal
  geometry_msgs::msg::PoseStamped subgoal = global_plan_.poses[subgoal_idx];
  subgoal.header.stamp = clock_->now();
  subgoal_pub_->publish(subgoal);
  
  if (subgoal_idx == global_plan_.poses.size() - 1) {
    RCLCPP_DEBUG(
      logger_, "[ROSNAV_CONTROLLER] Published subgoal (end of plan): x=%.2f, y=%.2f, arc_dist=%.2f",
      subgoal.pose.position.x, subgoal.pose.position.y, accumulated_dist);
  } else {
    RCLCPP_DEBUG(
      logger_, "[ROSNAV_CONTROLLER] Published subgoal (lookahead): x=%.2f, y=%.2f, arc_dist=%.2f (target: %.2f)",
      subgoal.pose.position.x, subgoal.pose.position.y, accumulated_dist, lookahead_dist);
  }
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
