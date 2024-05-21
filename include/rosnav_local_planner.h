#ifndef ROSNAV_LOCAL_PLANNER_H
#define ROSNAV_LOCAL_PLANNER_H

#include <ros/ros.h>

// base local planner base class and utilities
#include <nav_core/base_local_planner.h>
#include <mbf_costmap_core/costmap_controller.h>
#include <base_local_planner/goal_functions.h>
#include <base_local_planner/odometry_helper_ros.h>

// transforms
#include <tf/transform_listener.h>

// costmap
#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_converter/costmap_converter_interface.h>

// dynamic reconfigure
#include <rosnav/RosnavLocalPlannerReconfigureConfig.h>
#include <dynamic_reconfigure/server.h>
// #include <rosnav/srv/GetAction.h>

#include <mbf_msgs/ExePathResult.h>

#include <rosnav_config.h>
#include <rosnav/GetAction.h>


namespace rosnav
{
    class RosnavLocalPlanner : public nav_core::BaseLocalPlanner,  public mbf_costmap_core::CostmapController
    {
    public:
        /**
        * @brief Default constructor of the rosnav plugin
        */
        RosnavLocalPlanner();

        /**
        * @brief Destructor of rosnav plugin
        */
        ~RosnavLocalPlanner();

        /**
         * @brief  Constructs the local planner
         * @param name The name to give this instance of the local planner
         * @param tf A pointer to a transform listener
         * @param costmap_ros The cost map to use for assigning costs to local plans
         */
        void initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros);


        /**
        * @brief Initializes the rosn plugin
        * @param name The name of the instance
        * @param tf Pointer to a transform listener
        * @param costmap_ros Cost map representing occupied and free space
        */
        void initialize();

        /**
        * @brief Set the plan that the rosnav local planner is following
        * @param orig_global_plan The plan to pass to the local planner
        * @return True if the plan was updated successfully, false otherwise
        */
        bool setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan);

        /**
        * @brief Given the current position, orientation, and velocity of the robot, compute velocity commands to send to the base
        * @param cmd_vel Will be filled with the velocity command to be passed to the robot base
        * @return True if a valid trajectory was found, false otherwise
        */
        bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);

        /**
        * @brief Given the current position, orientation, and velocity of the robot, compute velocity commands to send to the base.
        * @remark Extended version for MBF API
        * @param pose the current pose of the robot.
        * @param velocity the current velocity of the robot.
        * @param cmd_vel Will be filled with the velocity command to be passed to the robot base.
        * @param message Optional more detailed outcome as a string
        * @return Result code as described on ExePath action result:
        *         SUCCESS         = 0
        *         1..9 are reserved as plugin specific non-error results
        *         FAILURE         = 100   Unspecified failure, only used for old, non-mfb_core based plugins
        *         CANCELED        = 101
        *         NO_VALID_CMD    = 102
        *         PAT_EXCEEDED    = 103
        *         COLLISION       = 104
        *         OSCILLATION     = 105
        *         ROBOT_STUCK     = 106
        *         MISSED_GOAL     = 107
        *         MISSED_PATH     = 108
        *         BLOCKED_PATH    = 109
        *         INVALID_PATH    = 110
        *         TF_ERROR        = 111
        *         NOT_INITIALIZED = 112
        *         INVALID_PLUGIN  = 113
        *         INTERNAL_ERROR  = 114
        *         121..149 are reserved as plugin specific errors
        */
        uint32_t computeVelocityCommands(const geometry_msgs::PoseStamped& pose, const geometry_msgs::TwistStamped& velocity,
                                        geometry_msgs::TwistStamped &cmd_vel, std::string &message);

        bool retrieveVelocityCommands(geometry_msgs::TwistStamped &cmd_vel);

        /**
        * @brief  Check if the goal pose has been achieved
        *
        * The actual check is performed in computeVelocityCommands().
        * Only the status flag is checked here.
        * @return True if achieved, false otherwise
        */
        bool isGoalReached();

        /**
            * @brief Dummy version to satisfy MBF API
            */
        bool isGoalReached(double xy_tolerance, double yaw_tolerance) { return isGoalReached(); };

        /**
        * @brief Requests the planner to cancel, e.g. if it takes too much time
        * @remark New on MBF API
        * @return True if a cancel has been successfully requested, false if not implemented.
        */
        bool cancel() { return false; };

        /**
        * @brief Callback for the dynamic_reconfigure node.
        * 
        * This callback allows to modify parameters dynamically at runtime without restarting the node
        * @param config Reference to the dynamic reconfigure config
        * @param level Dynamic reconfigure level
        */
        void reconfigureCB(RosnavLocalPlannerReconfigureConfig& config, uint32_t level);

        bool updateAgentSelection(std::string agent_name);

        bool getServiceClient(std::string service_name, ros::ServiceClient& client, double_t timeout);

        std::vector<std::string> translateStrToList(std::string str, std::string seperator);

        /**
         * @brief Saturate the translational and angular velocity to given limits.
         * 
         * The limit of the translational velocity for backwards driving can be changed independently.
         * Do not choose max_vel_x_backwards <= 0. If no backward driving is desired, change the optimization weight for
         * penalizing backwards driving instead.
         * @param[in,out] vx The translational velocity that should be saturated.
         * @param[in,out] vy Strafing velocity which can be nonzero for holonomic robots
         * @param[in,out] omega The angular velocity that should be saturated.
         * @param max_vel_x Maximum translational velocity for forward driving
         * @param max_vel_y Maximum strafing velocity (for holonomic robots)
         * @param max_vel_trans Maximum translational velocity for holonomic robots
         * @param max_vel_theta Maximum (absolute) angular velocity
         * @param max_vel_x_backwards Maximum translational velocity for backwards driving
         */
        void saturateVelocity(double& vx, double& vy, double& omega, double max_vel_x, double max_vel_y,
                                double max_vel_trans, double max_vel_theta, double max_vel_x_backwards) const;

        /**
         * @brief Return the internal config mutex
        */
        boost::mutex& configMutex() {return config_mutex_;}

    private:
        // local planner
        // boost::shared_ptr<rosnavLocalPlannerROS> rosnav_local_planner_;

        // dynamic reconfigure
        // dynamic_reconfigure::Server<rosnav_local_planner::RosnavLocalPlanner> *dsrv_;

        bool initialized_; //!< Keeps track about the correct initialization of this class
        
        // rosnav config
        RosnavConfig config_;

        ros::NodeHandle nh_; // ROS node handle
        ros::ServiceClient client_; // service client for cmd vel retrieval

        std::string active_agent;   // active agent name considered for cmd vel retrieval
        std::vector<std::string> agent_selection_list;  
        std::map<std::string, ros::ServiceClient> agent_srv_clients;

        boost::shared_ptr< dynamic_reconfigure::Server<RosnavLocalPlannerReconfigureConfig> > dynamic_recfg_; //!< Dynamic reconfigure server to allow config modifications at runtime
        boost::mutex config_mutex_; //!< Mutex for config accesses and changes

    };
}; // namespace rosnav_local_planner

#endif // ROSNAV_LOCAL_PLANNER_H

