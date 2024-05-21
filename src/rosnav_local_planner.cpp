#include <rosnav_local_planner.h>
// pluginlib macros
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>


// register this planner as a BaseLocalPlanner plugin
PLUGINLIB_EXPORT_CLASS(rosnav::RosnavLocalPlanner, nav_core::BaseLocalPlanner)
PLUGINLIB_EXPORT_CLASS(rosnav::RosnavLocalPlanner, mbf_costmap_core::CostmapController)

rosnav::RosnavLocalPlanner::RosnavLocalPlanner(): initialized_(false)
{
}

rosnav::RosnavLocalPlanner::~RosnavLocalPlanner()
{
}


void rosnav::RosnavLocalPlanner::initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros)
{
    if (!initialized_)
    {
        ROS_INFO("Initializing RosnavLocalPlanner...");

        // create Node handle
        ros::NodeHandle nh_("~/" + name);
        config_.loadRosParamFromNodeHandle(nh_);

        nh_.param("/agent_name", active_agent, std::string(""));

        if (!active_agent.empty())
        {
            updateAgentSelection(active_agent);
            client_ = agent_srv_clients[active_agent];
        }
        else
        {
            ROS_ERROR("No agent selected. Please provide an agent name in the parameter server or via dynamic reconfigure.");
        }

        // initialize dynamic reconfigure
        dynamic_recfg_ = boost::make_shared< dynamic_reconfigure::Server<RosnavLocalPlannerReconfigureConfig> >(nh_);
        dynamic_reconfigure::Server<RosnavLocalPlannerReconfigureConfig>::CallbackType cb = boost::bind(&RosnavLocalPlanner::reconfigureCB, this, _1, _2);
        dynamic_recfg_->setCallback(cb);

        initialized_ = true;
    }
}

bool rosnav::RosnavLocalPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan)
{
    // check if plugin is initialized
    if (!initialized_)
    {
        ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
        return false;
    }
    ROS_DEBUG("Setting plan...");
    return true;
}

bool rosnav::RosnavLocalPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
{
    // check if plugin is initialized
    if (!initialized_)
    {
        ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
        return false;
    }
    ROS_ERROR("Method not implemented yet!");
    // call service to retrieve cmd vel
    return false;
}

u_int32_t rosnav::RosnavLocalPlanner::computeVelocityCommands(const geometry_msgs::PoseStamped& pose, const geometry_msgs::TwistStamped& velocity,
                                        geometry_msgs::TwistStamped &cmd_vel, std::string &message)
{
    // check if plugin is initialized
    if (!initialized_)
    {
        ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
        return mbf_msgs::ExePathResult::INTERNAL_ERROR;
    }
    bool succ = retrieveVelocityCommands(cmd_vel);
    // saturateVelocity(
    //     cmd_vel.twist.linear.x, 
    //     cmd_vel.twist.linear.y, 
    //     cmd_vel.twist.angular.z, 
    //     config_.robot.max_vel_x, 
    //     config_.robot.max_vel_y, 
    //     config_.robot.max_vel_trans, 
    //     config_.robot.max_vel_theta, 
    //     config_.robot.max_vel_x_backwards
    // );
    return mbf_msgs::ExePathResult::SUCCESS;
}

bool rosnav::RosnavLocalPlanner::retrieveVelocityCommands(geometry_msgs::TwistStamped &cmd_vel)
{
    ROS_DEBUG("Retrieving velocity commands...");

    rosnav::GetAction srv;
    if (client_.call(srv))
    {
        cmd_vel.twist.linear.x = srv.response.action[0];
        cmd_vel.twist.linear.y = srv.response.action[1];
        cmd_vel.twist.angular.z = srv.response.action[2];
    }
    else
    {
        ROS_ERROR("Failed to call service '%s'", client_.getService().c_str());
        return false;
    }
    return true;
}


bool rosnav::RosnavLocalPlanner::isGoalReached()
{
    // check if plugin is initialized
    if (!initialized_)
    {
        ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
        return false;
    }
    ROS_DEBUG("isGoalReached(): Not implemented yet! (not needed for now)");
    return false;
}

void rosnav::RosnavLocalPlanner::reconfigureCB(RosnavLocalPlannerReconfigureConfig& config, uint32_t level)
{
    boost::mutex::scoped_lock l(config_mutex_);

    config_.reconfigure(config);

    // check if plugin is initialized
    if (!initialized_)
    {
        ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
        return;
    }

    // check if the agent has changed
    if (config.active_agent == active_agent)
    {
        return;
    }

    // deal with new agent
    if (!(std::find(agent_selection_list.begin(), agent_selection_list.end(), config.active_agent) != agent_selection_list.end())) {
        ROS_INFO("New agent detected: %s", config.active_agent.c_str());
        if (!(updateAgentSelection(config.active_agent)))
        {
            ROS_INFO("Keeping the current agent selection. Active agent '%s'", active_agent.c_str());
        }
    }

    // update the active agent
    active_agent = config.active_agent;
    client_ = agent_srv_clients[active_agent];

    config_mutex_.unlock();
}

std::vector<std::string> rosnav::RosnavLocalPlanner::translateStrToList(std::string str, std::string separator)
{
    std::vector<std::string> result;
    size_t pos = 0;
    while ((pos = str.find(separator)) != std::string::npos)
    {
        result.push_back(str.substr(0, pos));
        str.erase(0, pos + separator.length());
    }
    result.push_back(str);
    return result;
}


bool rosnav::RosnavLocalPlanner::updateAgentSelection(std::string agent_name)
{
    std::string service_name = "rosnav/" + agent_name + "/get_action";

    ros::ServiceClient client;
    if (getServiceClient(service_name, client, 10.0))
    {
        ROS_INFO("Service '%s' is online. Updating the agent selection list.", service_name.c_str());
        agent_srv_clients.insert({agent_name, client});
        agent_selection_list.push_back(agent_name);

        ROS_INFO("Agent selection list:");
        for (const auto& agent : agent_selection_list) {
            ROS_INFO("- %s", agent.c_str());
        }
        return true;
    }

    ROS_ERROR("Service '%s' is not online. Check if the agent node is running. Is the service advertised in the agents' namespace?", service_name.c_str());
    return false;
}

bool rosnav::RosnavLocalPlanner::getServiceClient(std::string service_name, ros::ServiceClient& client, double_t timeout=10)
{
    // Wait for the service to become available
    if (ros::service::waitForService(service_name, ros::Duration(timeout)))
    {
        // initialize service clients to retrieve cmd vel on "rosnav/*agent_name*/get_action"
        client = nh_.serviceClient<rosnav::GetAction>(service_name);
        return true;
    }
    return false;

}

void rosnav::RosnavLocalPlanner::saturateVelocity(double& vx, double& vy, double& omega, double max_vel_x, double max_vel_y, double max_vel_trans, double max_vel_theta, double max_vel_x_backwards) const
{
    double ratio_x = 1, ratio_omega = 1, ratio_y = 1;
    // Limit translational velocity for forward driving
    if (vx > max_vel_x)
        ratio_x = max_vel_x / vx;
    
    // limit strafing velocity
    if (vy > max_vel_y || vy < -max_vel_y)
        ratio_y = std::abs(max_vel_y / vy);
    
    // Limit angular velocity
    if (omega > max_vel_theta || omega < -max_vel_theta)
        ratio_omega = std::abs(max_vel_theta / omega);
    
    // Limit backwards velocity
    if (max_vel_x_backwards<=0)
    {
        ROS_WARN_ONCE("RosnavLocalPlanner(): Do not choose max_vel_x_backwards to be <=0. Disable backwards driving by increasing the optimization weight for penalyzing backwards driving.");
    }
    else if (vx < -max_vel_x_backwards)
        ratio_x = - max_vel_x_backwards / vx;

    if (config_.robot.use_proportional_saturation)
    {
        double ratio = std::min(std::min(ratio_x, ratio_y), ratio_omega);
        vx *= ratio;
        vy *= ratio;
        omega *= ratio;
    }
    else
    {
        vx *= ratio_x;
        vy *= ratio_y;
        omega *= ratio_omega;
    }

    double vel_linear = std::hypot(vx, vy);
    if (vel_linear > max_vel_trans)
    {
        double max_vel_trans_ratio = max_vel_trans / vel_linear;
        vx *= max_vel_trans_ratio;
        vy *= max_vel_trans_ratio;
    }
}