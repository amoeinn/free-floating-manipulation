// Does an arm trajectory stay collision free once the base reacts to it?
//
// A fixed base planner answers a different question from the one a free
// flying servicer needs answered. It checks the arm against the world at
// each configuration, holding the base still. On a free floating base the
// base does not stay still: momentum is conserved, so every joint motion
// pushes it, and where the arm actually is depends on the whole history of
// the trajectory rather than on the current joint angles.
//
// This walks a trajectory, integrates the base motion the dynamics imply,
// writes the result into a floating root joint, and asks the planning scene
// about the corrected state. The same trajectory is also checked with the
// base held at the identity, so the two answers can be compared directly and
// the difference attributed to base reaction rather than to anything else.

#ifndef FREE_FLOATING_MANIPULATION_TRAJECTORY_VALIDITY_HPP
#define FREE_FLOATING_MANIPULATION_TRAJECTORY_VALIDITY_HPP

#include "free_floating_manipulation/dynamics.hpp"

#include <Eigen/Geometry>
#include <moveit/planning_scene/planning_scene.hpp>

#include <memory>
#include <string>
#include <vector>

namespace free_floating_manipulation
{

/// Where a trajectory first stops being collision free, and by how much.
struct ValidityReport
{
  bool collides = false;
  /// Index of the first waypoint that collides.
  std::size_t waypoint = 0;
  /// That waypoint's position along the path, 0 at the start, 1 at the end.
  double path_fraction = 0.0;
  /// Deepest penetration at that waypoint, in metres.
  double penetration = 0.0;
  std::string body_a;
  std::string body_b;
  /// Base pose at that waypoint, relative to where the base started.
  Eigen::Isometry3d base_pose = Eigen::Isometry3d::Identity();
  /// Net base rotation over the whole trajectory, radians.
  double base_rotation = 0.0;
  /// How far the base centre of mass drifted over the whole trajectory.
  double base_translation = 0.0;
};

class TrajectoryValidityChecker
{
public:
  /// \param scene the world the arm is checked against
  /// \param dynamics supplies H_b and H_bm along the trajectory
  /// \param arm_joints the trajectory's joints, in column order
  /// \param virtual_joint the SRDF floating joint the integrated base pose is
  ///        written into. The RobotModel must declare one, or the base cannot
  ///        move and this reduces to the fixed base check it exists to beat.
  TrajectoryValidityChecker(const planning_scene::PlanningScenePtr& scene,
                            std::shared_ptr<const FloatingBaseModel> dynamics,
                            std::vector<std::string> arm_joints,
                            std::string virtual_joint);

  /// Base pose at each waypoint, integrated from zero momentum.
  ///
  /// The base twist is [linear, angular] in the base body frame, so it
  /// integrates as a body frame rate: the pose advances by its own
  /// exponential rather than by a world frame increment. Using the world
  /// form here gives an answer that is wrong by a factor of several and
  /// still looks like a pose.
  std::vector<Eigen::Isometry3d> integrateBase(
      const std::vector<std::vector<double>>& waypoints, double duration) const;

  /// Check every waypoint with the base held fixed at the identity.
  ValidityReport checkFixedBase(
      const std::vector<std::vector<double>>& waypoints) const;

  /// Check every waypoint with the base where the dynamics put it.
  ValidityReport checkFreeFloating(
      const std::vector<std::vector<double>>& waypoints, double duration) const;

private:
  ValidityReport walk(const std::vector<std::vector<double>>& waypoints,
                      const std::vector<Eigen::Isometry3d>& base_poses) const;
  void applyState(moveit::core::RobotState& state,
                  const std::vector<double>& angles,
                  const Eigen::Isometry3d& base_pose) const;

  planning_scene::PlanningScenePtr scene_;
  std::shared_ptr<const FloatingBaseModel> dynamics_;
  std::vector<std::string> arm_joints_;
  std::string virtual_joint_;
};

}  // namespace free_floating_manipulation

#endif  // FREE_FLOATING_MANIPULATION_TRAJECTORY_VALIDITY_HPP
