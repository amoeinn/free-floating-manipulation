// The mission executive: a behavior tree that sequences work already verified.
//
// Almost nothing here is new logic, and that is deliberate. Acquisition, the
// flip test and tracking are phase 4's, reached over a ROS 2 action and a
// service. The free-floating trajectory check is phase 3's C++ library,
// called directly. Planning is MoveIt's OMPL, set up as the phase 3
// demonstration sets it up. The tree's own contribution is the order, the
// abort conditions, and one thing the phase would be pointless without: a
// representation of a belief that can no longer be checked.
//
// That last part is why this file is not just a Sequence. Phase 4 established
// that a flipped acquisition is held indefinitely, smoothly and confidently,
// with nothing in the track reporting it, and that the only detector is valid
// solely at handover and inverts past 40 degrees of body frame sun motion. A
// tree that runs Acquire then Track and returns SUCCESS has thrown that away.
//
// BehaviorTree.CPP's alphabet is SUCCESS, FAILURE and RUNNING. "Succeeded, on
// a belief nothing can check any more" is not in it. So the provenance of the
// pose belief is carried on the blackboard and published in the mission
// report, and the audit node decides separately whether the mission was right.
// A tree returning SUCCESS is the same class of claim as a service returning
// true.

#include "free_floating_manipulation/dynamics.hpp"
#include "free_floating_manipulation/trajectory_validity.hpp"

#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/action_node.h>

#include <free_floating_manipulation/action/acquire_target.hpp>
#include <free_floating_manipulation/srv/verify_flip.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometric_shapes/shapes.h>
#include <moveit/collision_detection/collision_env.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <srdfdom/model.h>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <urdf_parser/urdf_parser.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

using free_floating_manipulation::FloatingBaseModel;
using free_floating_manipulation::TrajectoryValidityChecker;
using free_floating_manipulation::ValidityReport;
using AcquireTarget = free_floating_manipulation::action::AcquireTarget;
using VerifyFlip = free_floating_manipulation::srv::VerifyFlip;

namespace
{
const std::vector<std::string> kArmJoints = { "panda_joint1", "panda_joint2",
                                              "panda_joint3", "panda_joint4",
                                              "panda_joint5", "panda_joint6",
                                              "panda_joint7" };
constexpr const char* kGroup = "panda_arm";
constexpr const char* kVirtualJoint = "virtual_joint";
constexpr double kDuration = 4.0;
// The flip test is decisive at the sun angle the model was fitted at, is
// ambiguous by 28 degrees and inverts past 40. Fifteen leaves margin on the
// side where it still separates: measured 1.69x against 0.59x there.
constexpr double kFlipWindowDeg = 15.0;

/// Everything the tree's nodes share. Held by the tree, not by any node.
struct Mission
{
  rclcpp::Node::SharedPtr node;
  rclcpp_action::Client<AcquireTarget>::SharedPtr acquire;
  rclcpp::Client<VerifyFlip>::SharedPtr verify;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr command;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr report;
  std::shared_ptr<TrajectoryValidityChecker> checker;
  planning_scene::PlanningScenePtr scene;
  moveit::core::RobotModelPtr robot;
  std::unique_ptr<pluginlib::ClassLoader<planning_interface::PlannerManager>> loader;
  planning_interface::PlannerManagerPtr planner;

  std::vector<std::vector<double>> waypoints;
  std::string abort_reason;
  // The provenance of the pose belief, which is the thing the tree cannot say
  // in its return value.
  bool belief_verified = false;
  double belief_sun_angle_deg = std::numeric_limits<double>::quiet_NaN();
  double flip_ratio = std::numeric_limits<double>::quiet_NaN();
  bool belief_repaired = false;
  ValidityReport fixed_report;
  ValidityReport free_report;
  bool planned = false;
  bool executed = false;
  double bus_kg = 200.0;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr estimates;
  std::atomic<int> estimate_count{ 0 };
  int estimates_at_execute = 0;
};

Mission* g_mission = nullptr;

std::string readFile(const std::string& path)
{
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("cannot open " + path);
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

/// Acquire the client's pose from imagery, with no prior.
///
/// Stateful because it cannot be a tick: eight restarts is about 106 s on
/// CPU. Returning RUNNING is what lets the tree stay responsive while a
/// perception process that knows nothing about trees does the work.
class AcquireTargetPose : public BT::StatefulActionNode
{
public:
  AcquireTargetPose(const std::string& name, const BT::NodeConfig& config)
    : BT::StatefulActionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return { BT::InputPort<int>("restarts", 8, "restarts drawn from SO(3)"),
             BT::InputPort<int>("iterations", 120, "optimiser steps per restart"),
             BT::InputPort<bool>("inject_flip", false,
                                 "test hook: return the 180 degree flip") };
  }

  BT::NodeStatus onStart() override
  {
    auto& m = *g_mission;
    if (!m.acquire->wait_for_action_server(std::chrono::seconds(10)))
    {
      m.abort_reason = "perception's acquire_target action never appeared";
      return BT::NodeStatus::FAILURE;
    }
    AcquireTarget::Goal goal;
    goal.restarts = getInput<int>("restarts").value_or(8);
    goal.iterations = getInput<int>("iterations").value_or(120);
    goal.inject_flip = getInput<bool>("inject_flip").value_or(false);
    RCLCPP_INFO(m.node->get_logger(), "acquiring with %d restarts%s",
                goal.restarts, goal.inject_flip ? " (FLIP INJECTED, test only)" : "");
    future_ = m.acquire->async_send_goal(goal);
    stage_ = Stage::kSent;
    return BT::NodeStatus::RUNNING;
  }

  BT::NodeStatus onRunning() override
  {
    auto& m = *g_mission;
    if (stage_ == Stage::kSent)
    {
      if (future_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
        return BT::NodeStatus::RUNNING;
      auto handle = future_.get();
      if (!handle)
      {
        m.abort_reason = "perception rejected the acquisition goal";
        return BT::NodeStatus::FAILURE;
      }
      result_ = m.acquire->async_get_result(handle);
      stage_ = Stage::kWaiting;
      return BT::NodeStatus::RUNNING;
    }
    if (result_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
      return BT::NodeStatus::RUNNING;
    const auto wrapped = result_.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED || !wrapped.result->acquired)
    {
      m.abort_reason = "acquisition did not converge: " + wrapped.result->reason;
      return BT::NodeStatus::FAILURE;
    }
    RCLCPP_INFO(m.node->get_logger(), "acquired: loss %.4e after %d restarts",
                wrapped.result->final_loss, wrapped.result->restarts_used);
    return BT::NodeStatus::SUCCESS;
  }

  void onHalted() override {}

private:
  enum class Stage { kSent, kWaiting };
  Stage stage_ = Stage::kSent;
  std::shared_future<rclcpp_action::ClientGoalHandle<AcquireTarget>::SharedPtr> future_;
  std::shared_future<rclcpp_action::ClientGoalHandle<AcquireTarget>::WrappedResult> result_;
};

/// The one check that can catch a flipped acquisition, run where it works.
///
/// This is not a retry and not a fallback. It fails the mission, because the
/// alternative is committing to a belief that the rest of the mission cannot
/// re-examine: silhouette tracking holds a wrong pose exactly as stably as a
/// right one.
class RejectFlippedAcquisition : public BT::StatefulActionNode
{
public:
  RejectFlippedAcquisition(const std::string& name, const BT::NodeConfig& config)
    : BT::StatefulActionNode(name, config) {}

  static BT::PortsList providedPorts()
  {
    return { BT::InputPort<double>("max_sun_angle_deg", kFlipWindowDeg,
                                   "refuse to answer beyond this") };
  }

  BT::NodeStatus onStart() override
  {
    auto& m = *g_mission;
    if (!m.verify->wait_for_service(std::chrono::seconds(10)))
    {
      m.abort_reason = "perception's verify_flip service never appeared";
      return BT::NodeStatus::FAILURE;
    }
    auto request = std::make_shared<VerifyFlip::Request>();
    request->max_sun_angle_deg = getInput<double>("max_sun_angle_deg").value_or(kFlipWindowDeg);
    future_ = m.verify->async_send_request(request).future.share();
    return BT::NodeStatus::RUNNING;
  }

  BT::NodeStatus onRunning() override
  {
    auto& m = *g_mission;
    if (future_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
      return BT::NodeStatus::RUNNING;
    const auto response = future_.get();
    m.belief_sun_angle_deg = response->sun_angle_deg;
    m.flip_ratio = response->ratio;
    if (!response->valid)
    {
      // Refusing is the correct answer outside the window. A verdict taken
      // past 40 degrees would repair a correct pose into the flip.
      m.abort_reason = "the flip test could not be taken: " + response->reason;
      m.belief_verified = false;
      return BT::NodeStatus::FAILURE;
    }
    m.belief_verified = true;
    m.belief_repaired = response->repaired;
    RCLCPP_INFO(m.node->get_logger(),
                "flip test at %.1f deg sun: ratio %.2fx, %s",
                response->sun_angle_deg, response->ratio, response->reason.c_str());
    return BT::NodeStatus::SUCCESS;
  }

  void onHalted() override {}

private:
  std::shared_future<VerifyFlip::Response::SharedPtr> future_;
};

/// Plan the reach with MoveIt's OMPL. Fixed base, because that is the only
/// planner this project has: J_g is built and verified and nothing plans with
/// it yet. The free-floating part of the problem is handled by rejecting what
/// this returns, not by planning in the right space, and the mission report
/// says so rather than letting the tree imply otherwise.
class PlanArmTrajectory : public BT::SyncActionNode
{
public:
  PlanArmTrajectory(const std::string& name, const BT::NodeConfig& config)
    : BT::SyncActionNode(name, config) {}
  static BT::PortsList providedPorts() { return {}; }

  BT::NodeStatus tick() override
  {
    auto& m = *g_mission;
    moveit::core::RobotState start(m.robot);
    start.setToDefaultValues();
    const std::vector<double> home{ 0.0, -1.30, 0.0, -2.60, 0.0, 1.50, 0.0 };
    const std::vector<double> goal{ 0.0, 0.25, 0.0, -1.05, 0.0, 1.30, 0.0 };
    for (std::size_t j = 0; j < kArmJoints.size(); ++j)
      start.setJointPositions(kArmJoints[j], &home[j]);
    start.update();
    m.scene->setCurrentState(start);

    moveit::core::RobotState target(start);
    for (std::size_t j = 0; j < kArmJoints.size(); ++j)
      target.setJointPositions(kArmJoints[j], &goal[j]);
    target.update();

    planning_interface::MotionPlanRequest request;
    request.group_name = kGroup;
    request.allowed_planning_time = 10.0;
    request.num_planning_attempts = 1;
    moveit::core::robotStateToRobotStateMsg(start, request.start_state);
    request.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(
        target, m.robot->getJointModelGroup(kGroup)));

    planning_interface::MotionPlanResponse response;
    auto context = m.planner->getPlanningContext(m.scene, request, response.error_code);
    if (context)
      context->solve(response);
    if (!context || !response.trajectory)
    {
      m.abort_reason = "OMPL returned no trajectory";
      return BT::NodeStatus::FAILURE;
    }
    m.waypoints.clear();
    for (std::size_t k = 0; k < response.trajectory->getWayPointCount(); ++k)
    {
      std::vector<double> angles(kArmJoints.size());
      for (std::size_t j = 0; j < kArmJoints.size(); ++j)
        angles[j] = response.trajectory->getWayPoint(k).getVariablePosition(kArmJoints[j]);
      m.waypoints.push_back(angles);
    }
    m.planned = true;
    RCLCPP_INFO(m.node->get_logger(), "planned %zu waypoints, fixed base",
                m.waypoints.size());
    return BT::NodeStatus::SUCCESS;
  }
};

/// Phase 3's checker, as a gate. The fixed base pass is checked too, because
/// "the free floating check failed" only means something if the ordinary
/// check passed: that pairing is the entire phase 3 result.
class CheckTrajectory : public BT::SyncActionNode
{
public:
  CheckTrajectory(const std::string& name, const BT::NodeConfig& config)
    : BT::SyncActionNode(name, config) {}
  static BT::PortsList providedPorts()
  {
    return { BT::InputPort<bool>("free_floating", true,
                                 "check with the base free rather than fixed") };
  }

  BT::NodeStatus tick() override
  {
    auto& m = *g_mission;
    const bool free_floating = getInput<bool>("free_floating").value_or(true);
    if (m.waypoints.empty())
    {
      m.abort_reason = "nothing to check: no trajectory";
      return BT::NodeStatus::FAILURE;
    }
    if (free_floating)
    {
      m.free_report = m.checker->checkFreeFloating(m.waypoints, kDuration);
      if (m.free_report.collides)
      {
        std::ostringstream why;
        why << "the base reaction puts " << m.free_report.body_a << " into "
            << m.free_report.body_b << " by " << std::fixed << std::setprecision(2)
            << m.free_report.penetration * 1000.0 << " mm at path fraction "
            << std::setprecision(3) << m.free_report.path_fraction
            << ", on a trajectory the fixed base check passes";
        m.abort_reason = why.str();
        RCLCPP_ERROR(m.node->get_logger(), "%s", m.abort_reason.c_str());
        return BT::NodeStatus::FAILURE;
      }
      RCLCPP_INFO(m.node->get_logger(),
                  "free floating check clear, base moves %.3f deg %.1f mm",
                  m.free_report.base_rotation * 180.0 / M_PI,
                  m.free_report.base_translation * 1000.0);
    }
    else
    {
      m.fixed_report = m.checker->checkFixedBase(m.waypoints);
      if (m.fixed_report.collides)
      {
        m.abort_reason = "the trajectory collides even with the base held still";
        return BT::NodeStatus::FAILURE;
      }
      RCLCPP_INFO(m.node->get_logger(), "fixed base check clear");
    }
    return BT::NodeStatus::SUCCESS;
  }
};

/// Hand the trajectory to the scene. This is the irreversible step, so it
/// asserts the pose belief's provenance rather than assuming the earlier
/// nodes left it good.
class ExecuteTrajectory : public BT::SyncActionNode
{
public:
  ExecuteTrajectory(const std::string& name, const BT::NodeConfig& config)
    : BT::SyncActionNode(name, config) {}
  static BT::PortsList providedPorts() { return {}; }

  BT::NodeStatus tick() override
  {
    auto& m = *g_mission;
    if (!m.belief_verified)
    {
      m.abort_reason = "refusing to execute on a pose belief that was never "
                       "verified inside the flip test's window";
      return BT::NodeStatus::FAILURE;
    }
    trajectory_msgs::msg::JointTrajectory msg;
    msg.joint_names = kArmJoints;
    for (std::size_t k = 0; k < m.waypoints.size(); ++k)
    {
      trajectory_msgs::msg::JointTrajectoryPoint point;
      point.positions = m.waypoints[k];
      const double t = kDuration * static_cast<double>(k) /
                       std::max<std::size_t>(m.waypoints.size() - 1, 1);
      point.time_from_start.sec = static_cast<int>(t);
      point.time_from_start.nanosec =
          static_cast<uint32_t>((t - static_cast<int>(t)) * 1e9);
      msg.points.push_back(point);
    }
    m.estimates_at_execute = m.estimate_count.load();
    m.command->publish(msg);
    m.executed = true;
    RCLCPP_INFO(m.node->get_logger(), "trajectory sent, %zu points over %.1f s",
                msg.points.size(), kDuration);
    return BT::NodeStatus::SUCCESS;
  }
};

/// Say what happened, including the part the return value cannot carry.
class ReportMission : public BT::SyncActionNode
{
public:
  ReportMission(const std::string& name, const BT::NodeConfig& config)
    : BT::SyncActionNode(name, config) {}
  static BT::PortsList providedPorts() { return {}; }

  BT::NodeStatus tick() override
  {
    auto& m = *g_mission;
    // Do not report until the mission's own pose belief is observable.
    //
    // The first silhouette step takes about 3.5 s, so a report published
    // immediately after execution beat the first estimate onto the wire and
    // the audit had nothing to judge. Worse, with an auditor that kept state
    // the verdict then landed on the previous mission's belief. A mission
    // that has not yet published what it believes has not finished.
    if (m.executed)
    {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
      while (rclcpp::ok() && m.estimate_count.load() <= m.estimates_at_execute &&
             std::chrono::steady_clock::now() < deadline)
        rclcpp::sleep_for(std::chrono::milliseconds(200));
      if (m.estimate_count.load() <= m.estimates_at_execute)
        RCLCPP_WARN(m.node->get_logger(),
                    "no fresh pose estimate within 60 s; the audit will have "
                    "nothing from this mission to judge");
      else
        RCLCPP_INFO(m.node->get_logger(),
                    "pose belief is being published; reporting now");
    }
    const bool ok = m.executed && m.abort_reason.empty();
    std::ostringstream json;
    json << std::boolalpha << "{"
         << "\"status\": \"" << (ok ? "SUCCESS" : "ABORT") << "\", "
         << "\"planned\": " << m.planned << ", "
         << "\"executed\": " << m.executed << ", "
         << "\"belief_verified\": " << m.belief_verified << ", "
         << "\"belief_repaired\": " << m.belief_repaired << ", "
         << "\"belief_sun_angle_deg\": " << std::fixed << std::setprecision(2)
         << m.belief_sun_angle_deg << ", "
         << "\"flip_ratio\": " << m.flip_ratio << ", "
         << "\"bus_kg\": " << std::setprecision(0) << m.bus_kg << ", "
         << std::setprecision(2)
         << "\"planner\": \"OMPL, fixed base; the free floating part is a "
            "rejection and not a plan\", "
         << "\"abort_reason\": \"" << m.abort_reason << "\", "
         << "\"caveat\": \"SUCCESS here means every node returned SUCCESS. It "
            "is not evidence the mission was right: after handover the pose "
            "belief cannot be rechecked, so see /mission/audit\"}";
    std_msgs::msg::String msg;
    msg.data = json.str();
    m.report->publish(msg);
    RCLCPP_INFO(m.node->get_logger(), "mission report: %s", msg.data.c_str());
    // Give the audit node a moment to receive it before the process exits.
    rclcpp::sleep_for(std::chrono::milliseconds(500));
    return BT::NodeStatus::SUCCESS;
  }
};

const char* kTreeXml = R"(
<root BTCPP_format="4">
  <BehaviorTree ID="Servicing">
    <Sequence>
      <Fallback>
        <Sequence name="mission">
          <AcquireTargetPose restarts="{restarts}" iterations="{iterations}"
                             inject_flip="{inject_flip}"/>
          <RejectFlippedAcquisition max_sun_angle_deg="15.0"/>
          <PlanArmTrajectory/>
          <CheckTrajectory free_floating="false"/>
          <CheckTrajectory free_floating="true"/>
          <ExecuteTrajectory/>
        </Sequence>
        <AlwaysSuccess name="abort_path"/>
      </Fallback>
      <ReportMission/>
    </Sequence>
  </BehaviorTree>
</root>
)";

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    std::cerr << "usage: mission_executive <urdf> <srdf> [restarts] [inject_flip] [bus_kg]\n";
    return 2;
  }
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("mission_executive");

  Mission mission;
  g_mission = &mission;
  mission.node = node;
  mission.acquire = rclcpp_action::create_client<AcquireTarget>(node, "acquire_target");
  mission.verify = node->create_client<VerifyFlip>("verify_flip");
  mission.command = node->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/servicer/joint_command", 10);
  mission.report = node->create_publisher<std_msgs::msg::String>("/mission/report", 10);
  mission.estimates = node->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/target/pose_estimate", 10,
      [](const geometry_msgs::msg::PoseStamped::SharedPtr) {
        g_mission->estimate_count.fetch_add(1);
      });

  auto urdf_model = urdf::parseURDF(readFile(argv[1]));
  auto srdf_model = std::make_shared<srdf::Model>();
  srdf_model->initString(*urdf_model, readFile(argv[2]));
  mission.robot = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
  mission.scene = std::make_shared<planning_scene::PlanningScene>(mission.robot);

  // The same obstacles the phase 3 demonstration used, so the free floating
  // gate is exercised on a scene where it is known to bite.
  auto add_box = [&](const std::string& name, const Eigen::Vector3d& half,
                     const Eigen::Vector3d& centre) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = centre;
    mission.scene->getWorldNonConst()->addToObject(
        name, shapes::ShapeConstPtr(new shapes::Box(2 * half.x(), 2 * half.y(),
                                                    2 * half.z())), pose);
  };
  add_box("servicer_body", { 0.30, 0.30, 0.18 }, { 0.0, 0.0, -0.32 });
  add_box("target_structure", { 0.26, 0.30, 0.06 }, { 0.46, 0.0, 0.96 });

  // The bus mass decides whether the free floating gate bites. Phase 3
  // measured 30 of 40 plans colliding at 200 kg and 8 of 40 at 2300, so
  // this is the difference between a mission that aborts correctly and
  // one that reaches the end, and both are worth being able to run.
  const double bus_mass = (argc > 5) ? std::stod(argv[5]) : 200.0;
  const double inertia = bus_mass * 0.25;
  auto dynamics = std::make_shared<FloatingBaseModel>(
      mission.robot, kArmJoints, "panda_hand", bus_mass,
      Eigen::Vector3d(inertia, inertia, inertia));
  mission.bus_kg = bus_mass;
  mission.checker = std::make_shared<TrajectoryValidityChecker>(
      mission.scene, dynamics, kArmJoints, kVirtualJoint);

  mission.loader = std::make_unique<pluginlib::ClassLoader<planning_interface::PlannerManager>>(
      "moveit_core", "planning_interface::PlannerManager");
  mission.planner = mission.loader->createUniqueInstance("ompl_interface/OMPLPlanner");
  if (!mission.planner->initialize(mission.robot, node, ""))
  {
    RCLCPP_ERROR(node->get_logger(), "the OMPL planner would not initialise");
    return 1;
  }

  BT::BehaviorTreeFactory factory;
  factory.registerNodeType<AcquireTargetPose>("AcquireTargetPose");
  factory.registerNodeType<RejectFlippedAcquisition>("RejectFlippedAcquisition");
  factory.registerNodeType<PlanArmTrajectory>("PlanArmTrajectory");
  factory.registerNodeType<CheckTrajectory>("CheckTrajectory");
  factory.registerNodeType<ExecuteTrajectory>("ExecuteTrajectory");
  factory.registerNodeType<ReportMission>("ReportMission");

  auto blackboard = BT::Blackboard::create();
  blackboard->set("restarts", (argc > 3) ? std::stoi(argv[3]) : 8);
  blackboard->set("iterations", 120);
  blackboard->set("inject_flip", (argc > 4) && std::string(argv[4]) == "true");
  auto tree = factory.createTreeFromText(kTreeXml, blackboard);

  std::thread spinner([node]() { rclcpp::spin(node); });
  BT::NodeStatus status = BT::NodeStatus::RUNNING;
  while (rclcpp::ok() && status == BT::NodeStatus::RUNNING)
  {
    status = tree.tickOnce();
    tree.sleep(std::chrono::milliseconds(100));
  }
  RCLCPP_INFO(node->get_logger(), "tree finished with %s",
              status == BT::NodeStatus::SUCCESS ? "SUCCESS" : "FAILURE");
  rclcpp::shutdown();
  spinner.join();
  return mission.abort_reason.empty() ? 0 : 3;
}
