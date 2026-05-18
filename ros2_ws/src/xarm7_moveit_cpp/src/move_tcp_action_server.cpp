#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include "xarm7_moveit_msgs/action/move_tcp.hpp"
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/collision_detection/collision_common.h>
#include <geometry_msgs/msg/pose.hpp>
#include <vector>
#include <thread>
#include <controller_manager_msgs/srv/list_controllers.hpp>
#include <mutex>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_spline_parameterization.h>


using MoveTCP = xarm7_moveit_msgs::action::MoveTCP;
using GoalHandleMoveTCP = rclcpp_action::ServerGoalHandle<MoveTCP>;

bool waitForController(
  const rclcpp::Node::SharedPtr& node,
  const std::string& controller_name,
  double timeout_sec = 10.0)
{
  auto client =
    node->create_client<controller_manager_msgs::srv::ListControllers>(
      "/controller_manager/list_controllers");

  if (!client->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_ERROR(node->get_logger(),
      "controller_manager service not available");
    return false;
  }

  auto start = node->now();

  while ((node->now() - start).seconds() < timeout_sec) {
    auto req =
      std::make_shared<controller_manager_msgs::srv::ListControllers::Request>();

    auto future = client->async_send_request(req);
    if (future.wait_for(std::chrono::milliseconds(200))
        == std::future_status::ready)
    {
      auto res = future.get();
      for (auto& c : res->controller) {
        if (c.name == controller_name && c.state == "active") {
          return true;
        }
      }
    }
  }
  return false;
}

class MoveTCPActionServer
  : public rclcpp::Node
{
public:
  MoveTCPActionServer()
  : Node("move_tcp_action_server")
  {
    action_server_ = rclcpp_action::create_server<MoveTCP>(
      this,
      "move_tcp",
      std::bind(&MoveTCPActionServer::handle_goal, this,
                std::placeholders::_1, std::placeholders::_2),
      std::bind(&MoveTCPActionServer::handle_cancel, this,
                std::placeholders::_1),
      std::bind(&MoveTCPActionServer::handle_accepted, this,
                std::placeholders::_1)
    );

    RCLCPP_INFO(get_logger(), "MoveTCP Action Server READY");
  }

private:
  rclcpp_action::Server<MoveTCP>::SharedPtr action_server_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::mutex move_group_mutex_;

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const MoveTCP::Goal> goal)
  {
    RCLCPP_INFO(get_logger(), "Received goal dx=%.3f", goal->dx);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleMoveTCP>)
  {
    RCLCPP_WARN(get_logger(), "Goal canceled");
    std::lock_guard<std::mutex> lock(move_group_mutex_);
    if (move_group_) {
      move_group_->stop();
    }
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(
    const std::shared_ptr<GoalHandleMoveTCP> goal_handle)
  {
    std::thread{
      std::bind(&MoveTCPActionServer::execute, this, goal_handle)
    }.detach();
  }

  void execute(const std::shared_ptr<GoalHandleMoveTCP> goal_handle)
  {
    auto node = shared_from_this();
    const auto goal = goal_handle->get_goal();
    auto result = std::make_shared<MoveTCP::Result>();

    RCLCPP_INFO(get_logger(), "Execute MoveTCP dx=%.3f", goal->dx);

    // ===== ここに move_tcp_x.cpp の main() 中身を入れる ====
    RCLCPP_INFO(node->get_logger(), "move_tcp_minus_x STARTED");
    move_group_ =
      std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        node, "xarm7");
    move_group_->setEndEffectorLink("tcp");
    move_group_->setPoseReferenceFrame("link_base");

    move_group_->setMaxVelocityScalingFactor(0.3);
    move_group_->setMaxAccelerationScalingFactor(0.3);

      // ===== プランニング条件（RViz相当）=====
    move_group_->setPlanningPipelineId("ompl");
    move_group_->setPlannerId("RRTConnectConfigDefault"); // or PRMstarkConfigDefault
    move_group_->setPlanningTime(5.0);
    move_group_->setNumPlanningAttempts(10);

    move_group_->setStartStateToCurrentState();

    move_group_->clearPoseTargets();

    // 現在 TCP 姿勢を取得
    auto current_pose_stamped = move_group_->getCurrentPose("tcp");
    auto target = current_pose_stamped.pose;

   
    target.position.x -= goal->dx; //0.2675;
    move_group_->setGoalPositionTolerance(0.01);      // 5mm
    move_group_->setGoalOrientationTolerance(0.005);   // 約0.6°

    // ===== ゴールIKを「現在姿勢に最も近いもの」に固定 =====
    auto current_state = *move_group_->getCurrentState();
    moveit::core::RobotState best_goal_state = current_state;
    const auto* jmg = current_state.getJointModelGroup("xarm7");
    auto psm = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
      node,
      "robot_description"
    );

    psm->startSceneMonitor();
    psm->startStateMonitor();
    psm->startWorldGeometryMonitor();

    auto planning_scene = psm->getPlanningScene();

    if (!planning_scene) {
      result->success = false;
      result->message = "PlanningScene not available";
      goal_handle->abort(result);
      return;
    }

    bool found_ik = false;
    double selected_dy = 0.0; 
    std::vector<double> y_offsets = {
      0.0,    // -x のみ
    -0.02,   // -x, -y 2cm
    -0.04,   // -x, -y 4cm
    -0.06    // -x, -y 6cm
    };

    for (double dy : y_offsets) {

      geometry_msgs::msg::Pose trial_target = target;
      trial_target.position.y += dy;

      RCLCPP_INFO(node->get_logger(),
        "Try IK with y offset = %.3f m", dy);

      int ik_trial_count = 0;
      for (int i = 0; i < 50; ++i) {
        if (goal_handle->is_canceling()) {
          result->success = false;
          result->message = "Canceled during IK search";
          goal_handle->canceled(result);
          return;
        }        
        ik_trial_count++;
        moveit::core::RobotState tmp_state = current_state;

        // seed を現在姿勢に固定
        tmp_state.setVariablePositions(
          current_state.getVariablePositions()
        );

        bool ok = tmp_state.setFromIK(
          jmg,
          trial_target,
          "tcp",
          0.0
        );

        if (!ok) continue;

        // ===== 衝突チェック =====
        collision_detection::CollisionRequest req;
        collision_detection::CollisionResult res;
        planning_scene->checkCollision(req, res, tmp_state);

        if (res.collision) {
          continue;   // 衝突 → 次のIK
        }

        // ===== 非衝突IKを発見 =====
        best_goal_state = tmp_state;
        found_ik = true;
        selected_dy = dy;  
        RCLCPP_INFO(node->get_logger(),
          "IK found after %d trials (y offset = %.3f m)",
          ik_trial_count, dy);
        break;
      }

      if (found_ik) break;  // 次の y 探索に行かない
    }


    if (found_ik) {
        RCLCPP_INFO(node->get_logger(),
            "Selected Y offset = %.3f m (%.0f mm)",
            selected_dy,
            selected_dy * 1000.0);
    }

    if (!found_ik) {
      result->success = false;
      result->message = "No suitable IK solution found";
      goal_handle->abort(result);
      return;
    }
    // ★ ここで関節目標を確定
    move_group_->setJointValueTarget(best_goal_state);



    double best_cost = 1e9;
    moveit::planning_interface::MoveGroupInterface::Plan best_plan;
    bool found = false;

    for (int i = 0; i < 10; ++i) {
      if (goal_handle->is_canceling()) {
        result->success = false;
        result->message = "Canceled during planning";
        goal_handle->canceled(result);
        return;
      }
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      auto res = move_group_->plan(plan);

      if (res == moveit::core::MoveItErrorCode::SUCCESS) {

        double cost = 0.0;
        double max_jump = 0.0;

        const auto& pts = plan.trajectory_.joint_trajectory.points;

        for (size_t j = 1; j < pts.size(); ++j) {
          for (size_t k = 0; k < pts[j].positions.size(); ++k) {
            double d = std::abs(
              pts[j].positions[k] - pts[j-1].positions[k]);
            cost += d;
            max_jump = std::max(max_jump, d);
          }
        }

        // ★ 一瞬のぶん回しを強く罰する
        cost += 5.0 * max_jump;


        if (cost < best_cost) {
          best_cost = cost;
          best_plan = plan;
          found = true;
        }
      }
    }

    if (found) {

      robot_trajectory::RobotTrajectory rt(
        move_group_->getRobotModel(),
        "xarm7"
      );

      rt.setRobotTrajectoryMsg(
        *move_group_->getCurrentState(),
        best_plan.trajectory_
      );

      trajectory_processing::IterativeSplineParameterization isp;
      isp.computeTimeStamps(rt, 1.0, 1.0);

      robot_trajectory::RobotTrajectory dense_rt(
          move_group_->getRobotModel(),
          "xarm7"
      );

      double dt = 0.01;
      double total = rt.getDuration();

      for (double t = 0.0; t <= total; t += dt)
      {
          moveit::core::RobotStatePtr state(
              new moveit::core::RobotState(rt.getRobotModel())
          );

          rt.getStateAtDurationFromStart(t, state);

          dense_rt.addSuffixWayPoint(*state, dt);
      }

      // 最終点保証
      dense_rt.addSuffixWayPoint(rt.getLastWayPoint(), 0.0);

      dense_rt.getRobotTrajectoryMsg(best_plan.trajectory_);



      // ★ executeしない

      result->trajectory = best_plan.trajectory_.joint_trajectory;
      result->success = true;
      result->message = "Trajectory planned only";
      goal_handle->succeed(result);
    }



  }
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveTCPActionServer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
