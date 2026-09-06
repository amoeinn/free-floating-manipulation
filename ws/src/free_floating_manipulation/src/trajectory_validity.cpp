#include "free_floating_manipulation/trajectory_validity.hpp"

#include <moveit/collision_detection/collision_common.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace free_floating_manipulation
{
namespace
{

/// Rotation matrix for an axis times angle vector, by Rodrigues.
Eigen::Matrix3d exponential(const Eigen::Vector3d& vector)
{
  const double angle = vector.norm();
  if (angle < 1e-15)
    return Eigen::Matrix3d::Identity();
  return Eigen::AngleAxisd(angle, vector / angle).toRotationMatrix();
}

}  // namespace

TrajectoryValidityChecker::TrajectoryValidityChecker(
    const planning_scene::PlanningScenePtr& scene,
    std::shared_ptr<const FloatingBaseModel> dynamics,
    std::vector<std::string> arm_joints, std::string virtual_joint)
  : scene_(scene)
  , dynamics_(std::move(dynamics))
  , arm_joints_(std::move(arm_joints))
  , virtual_joint_(std::move(virtual_joint))
{
  const moveit::core::JointModel* joint =
      scene_->getRobotModel()->getJointModel(virtual_joint_);
  if (joint == nullptr)
    throw std::runtime_error("no joint named " + virtual_joint_ +
                             "; the SRDF must declare a floating virtual joint "
                             "or the base cannot move");
  if (joint->getType() != moveit::core::JointModel::FLOATING)
    throw std::runtime_error(virtual_joint_ + " is not a floating joint, so the "
                             "integrated base pose has nowhere to go");
}

std::vector<Eigen::Isometry3d> TrajectoryValidityChecker::integrateBase(
    const std::vector<std::vector<double>>& waypoints, double duration) const
{
  std::vector<Eigen::Isometry3d> poses;
  poses.reserve(waypoints.size());
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  poses.push_back(pose);
  if (waypoints.size() < 2)
    return poses;

  moveit::core::RobotState state(scene_->getRobotModel());
  state.setToDefaultValues();
  const double step = duration / static_cast<double>(waypoints.size() - 1);
  const Eigen::Index n = static_cast<Eigen::Index>(arm_joints_.size());

  for (std::size_t k = 0; k + 1 < waypoints.size(); ++k)
  {
    // Joint rates from the waypoint spacing, and the coupling evaluated at
    // the midpoint of the segment rather than at either end.
    Eigen::VectorXd rates(n);
    std::vector<double> midpoint(arm_joints_.size());
    for (std::size_t j = 0; j < arm_joints_.size(); ++j)
    {
      rates[static_cast<Eigen::Index>(j)] =
          (waypoints[k + 1][j] - waypoints[k][j]) / step;
      midpoint[j] = 0.5 * (waypoints[k][j] + waypoints[k + 1][j]);
    }
    for (std::size_t j = 0; j < arm_joints_.size(); ++j)
      state.setJointPositions(arm_joints_[j], &midpoint[j]);
    state.update();

    const Eigen::Matrix<double, 6, 1> twist = dynamics_->baseVelocity(state, rates);

    // Body frame rates: advance the pose by its own exponential.
    pose.translation() += pose.linear() * twist.head<3>() * step;
    pose.linear() = pose.linear() * exponential(twist.tail<3>() * step);
    poses.push_back(pose);
  }
  return poses;
}

void TrajectoryValidityChecker::applyState(moveit::core::RobotState& state,
                                           const std::vector<double>& angles,
                                           const Eigen::Isometry3d& base_pose) const
{
  for (std::size_t j = 0; j < arm_joints_.size(); ++j)
    state.setJointPositions(arm_joints_[j], &angles[j]);

  const Eigen::Quaterniond rotation(base_pose.linear());
  const double values[7] = { base_pose.translation().x(),
                             base_pose.translation().y(),
                             base_pose.translation().z(),
                             rotation.x(), rotation.y(),
                             rotation.z(), rotation.w() };
  state.setJointPositions(virtual_joint_, values);
  state.update();
}

ValidityReport TrajectoryValidityChecker::walk(
    const std::vector<std::vector<double>>& waypoints,
    const std::vector<Eigen::Isometry3d>& base_poses) const
{
  ValidityReport report;
  if (!base_poses.empty())
  {
    const Eigen::Isometry3d& last = base_poses.back();
    report.base_rotation = Eigen::AngleAxisd(last.linear()).angle();
    report.base_translation = last.translation().norm();
  }

  moveit::core::RobotState state(scene_->getRobotModel());
  state.setToDefaultValues();

  collision_detection::CollisionRequest request;
  request.contacts = true;
  request.max_contacts = 32;
  request.max_contacts_per_pair = 4;
  request.distance = false;

  for (std::size_t k = 0; k < waypoints.size(); ++k)
  {
    applyState(state, waypoints[k], base_poses[k]);

    collision_detection::CollisionResult result;
    scene_->checkCollision(request, result, state);
    if (!result.collision)
      continue;

    report.collides = true;
    report.waypoint = k;
    report.path_fraction = waypoints.size() < 2
        ? 0.0
        : static_cast<double>(k) / static_cast<double>(waypoints.size() - 1);
    report.base_pose = base_poses[k];

    // Deepest contact at this waypoint, which is the number worth reporting:
    // a grazing touch and a limb through a wall are both "collision".
    for (const auto& pair : result.contacts)
      for (const collision_detection::Contact& contact : pair.second)
        if (contact.depth > report.penetration)
        {
          report.penetration = contact.depth;
          report.body_a = contact.body_name_1;
          report.body_b = contact.body_name_2;
        }
    return report;
  }
  return report;
}

ValidityReport TrajectoryValidityChecker::checkFixedBase(
    const std::vector<std::vector<double>>& waypoints) const
{
  const std::vector<Eigen::Isometry3d> fixed(waypoints.size(),
                                             Eigen::Isometry3d::Identity());
  return walk(waypoints, fixed);
}

ValidityReport TrajectoryValidityChecker::checkFreeFloating(
    const std::vector<std::vector<double>>& waypoints, double duration) const
{
  return walk(waypoints, integrateBase(waypoints, duration));
}

}  // namespace free_floating_manipulation
