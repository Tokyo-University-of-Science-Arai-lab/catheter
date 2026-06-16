#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <thread>
#include <controller_manager_msgs/srv/list_controllers.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/collision_detection/collision_common.h>

// ============================
// MoveItErrorCode を文字列化
// ============================
std::string errorCodeToString(const moveit::core::MoveItErrorCode& code)
{
  switch (code.val) {
    case moveit::core::MoveItErrorCode::SUCCESS: return "SUCCESS";
    case moveit::core::MoveItErrorCode::FAILURE: return "FAILURE";
    case moveit::core::MoveItErrorCode::PLANNING_FAILED: return "PLANNING_FAILED";
    case moveit::core::MoveItErrorCode::INVALID_MOTION_PLAN: return "INVALID_MOTION_PLAN";
    case moveit::core::MoveItErrorCode::CONTROL_FAILED: return "CONTROL_FAILED";
    case moveit::core::MoveItErrorCode::TIMED_OUT: return "TIMED_OUT";
    default: return "UNKNOWN_ERROR";
  }
}

// ============================
// controller 待ち
// ============================
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

    // executor は main で回っているので spin しない
    while (future.wait_for(std::chrono::milliseconds(100))
           != std::future_status::ready)
    {
      rclcpp::sleep_for(std::chrono::milliseconds(50));
    }

    auto res = future.get();
    for (auto& c : res->controller) {
      if (c.name == controller_name && c.state == "active") {
        RCLCPP_INFO(node->get_logger(),
          "Controller [%s] is ACTIVE", controller_name.c_str());
        return true;
      }
    }
  }

  RCLCPP_ERROR(node->get_logger(),
    "Timeout waiting for controller [%s]",
    controller_name.c_str());
  return false;
}

// ============================
// main
// ============================
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("move_tcp_minus_x");

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  RCLCPP_INFO(node->get_logger(), "move_tcp_minus_x STARTED");

  moveit::planning_interface::MoveGroupInterface move_group(node, "xarm7");
  move_group.setEndEffectorLink("tcp");
  move_group.setPoseReferenceFrame("link_base");

  if (!waitForController(node, "xarm7_traj_controller")) {
    RCLCPP_FATAL(node->get_logger(), "Controller not ready");
    rclcpp::shutdown();
    spin_thread.join();
    return 1;
  }

  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);

    // ===== プランニング条件（RViz相当）=====
  move_group.setPlanningPipelineId("ompl");
  move_group.setPlannerId("RRTConnectConfigDefault"); // or PRMstarkConfigDefault
  move_group.setPlanningTime(5.0);
  move_group.setNumPlanningAttempts(10);

  move_group.setStartStateToCurrentState();

  move_group.clearPoseTargets();

  // 現在 TCP 姿勢を取得
  auto current_pose_stamped = move_group.getCurrentPose("tcp");
  auto target = current_pose_stamped.pose;

  // 例：Z方向に +5cm 動かす
  target.position.x -= 0.2675; //0.2675;
  //target.position.y -= 0.05; //0.2675;

  move_group.setGoalPositionTolerance(0.01);      // 5mm
  move_group.setGoalOrientationTolerance(0.005);   // 約0.6°

  // ===== ゴールIKを「現在姿勢に最も近いもの」に固定 =====
  auto current_state = *move_group.getCurrentState();
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
    RCLCPP_FATAL(node->get_logger(),
      "PlanningScene is not available");
    rclcpp::shutdown();
    spin_thread.join();
    return 1;
  }

  bool found_ik = false;

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
      RCLCPP_INFO(node->get_logger(),
        "IK found after %d trials (y offset = %.3f m)",
        ik_trial_count, dy);
      break;
    }

    if (found_ik) break;  // 次の y 探索に行かない
  }


  if (!found_ik) {
    RCLCPP_ERROR(node->get_logger(), "No suitable IK solution found");
    rclcpp::shutdown();
    spin_thread.join();
    return 1;
  }

  // ★ ここで関節目標を確定
  move_group.setJointValueTarget(best_goal_state);



  double best_cost = 1e9;
  moveit::planning_interface::MoveGroupInterface::Plan best_plan;
  bool found = false;

  for (int i = 0; i < 10; ++i) {
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto res = move_group.plan(plan);

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
    RCLCPP_INFO(node->get_logger(),
      "Best joint motion cost = %.3f", best_cost);
    move_group.execute(best_plan);
  } else {
    RCLCPP_ERROR(node->get_logger(), "No valid plan found");
  }


  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
