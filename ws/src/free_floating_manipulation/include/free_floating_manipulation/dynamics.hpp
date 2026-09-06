// Floating base dynamics for a serial arm, read through MoveIt's RobotModel.
//
// A C++ port of the phase 2 torch implementation. With zero initial momentum
// and no external wrench the base twist is pinned by the joint rates,
//
//     H_b v_b + H_bm qdot = 0   ->   v_b = -H_b^-1 H_bm qdot
//
// and the end effector picks up that reaction,
//
//     xdot = (J_m - J_b H_b^-1 H_bm) qdot = J_g qdot.
//
// Conventions, chosen to match the Python implementation exactly so that the
// two can be compared block by block:
//
//   * the base twist is [linear, angular], not PyBullet's [angular, linear]
//   * it is written about the base link's inertial frame
//   * arm joints that are not ancestors of a link contribute zero columns to
//     that link's Jacobian, which the geometric form does not give for free
//
// Where MoveIt differs from PyBullet, and it does, the difference is measured
// by examples/verify_cpp_dynamics.py rather than assumed. MoveIt reports link
// frames directly through RobotState::getGlobalLinkTransform, and reads joint
// origins from the URDF as the spec defines them, relative to the parent link
// frame. PyBullet reports joint origins relative to the parent's inertial
// frame, which is an artifact of how Bullet stores links and which cost phase
// 1 several wrong theories. That artifact should not appear here.

#ifndef FREE_FLOATING_MANIPULATION_DYNAMICS_HPP
#define FREE_FLOATING_MANIPULATION_DYNAMICS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>

#include <string>
#include <vector>

namespace free_floating_manipulation
{

/// Skew symmetric matrix S with S w = v x w.
Eigen::Matrix3d skew(const Eigen::Vector3d& v);

/// One link's fixed inertial data, plus which arm joints actually move it.
struct LinkInertia
{
  const moveit::core::LinkModel* link = nullptr;
  std::string name;
  double mass = 0.0;
  /// Inertial frame relative to the link frame.
  Eigen::Isometry3d inertial_transform = Eigen::Isometry3d::Identity();
  /// Inertia about the centre of mass, in the inertial frame.
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
  /// Indices into the arm joint list. A proximal link is moved by only the
  /// first few, and the columns for the rest must be zero rather than the
  /// geometric formula's non zero garbage.
  std::vector<std::size_t> arm_ancestors;
};

/// World point and axis of one revolute joint, which is what a geometric
/// Jacobian column is built from.
struct JointFrame
{
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  Eigen::Vector3d axis = Eigen::Vector3d::Zero();
};

class FloatingBaseModel
{
public:
  /// \param model a RobotModel built from a URDF that carries inertial data.
  ///        MoveIt's own panda description does not; see the verification
  ///        script for what that means in practice.
  /// \param arm_joints the joints treated as generalized coordinates, in order
  /// \param end_effector_link the link J_g is written for
  /// \param base_mass if positive, replaces the base link's mass. This is
  ///        the spacecraft bus, which the URDF does not carry: the base link
  ///        of an arm model is a mounting flange, not a servicer.
  /// \param base_inertia_diagonal if all positive, replaces the base link's
  ///        inertia with that diagonal.
  FloatingBaseModel(const moveit::core::RobotModelConstPtr& model,
                    const std::vector<std::string>& arm_joints,
                    const std::string& end_effector_link,
                    double base_mass = -1.0,
                    const Eigen::Vector3d& base_inertia_diagonal =
                        Eigen::Vector3d::Constant(-1.0));

  std::size_t dofs() const { return arm_joints_.size(); }
  const std::vector<LinkInertia>& links() const { return links_; }
  double totalMass() const;

  /// World (point, axis) for each arm joint, in arm joint order.
  std::vector<JointFrame> armJointFrames(const moveit::core::RobotState& state) const;

  /// Linear and angular Jacobian of a point rigidly fixed to a link, base held
  /// fixed. Columns for joints outside \p ancestors are zero.
  void geometricColumns(const Eigen::Vector3d& target,
                        const std::vector<JointFrame>& frames,
                        const std::vector<std::size_t>& ancestors,
                        Eigen::Ref<Eigen::MatrixXd> linear,
                        Eigen::Ref<Eigen::MatrixXd> angular) const;

  /// World pose of a link's inertial frame.
  Eigen::Isometry3d comPose(const LinkInertia& link,
                            const moveit::core::RobotState& state) const;

  /// The point the base twist is written about: the base link inertial frame.
  Eigen::Vector3d referencePoint(const moveit::core::RobotState& state) const;

  /// (6 + n) x (6 + n) floating base mass matrix.
  Eigen::MatrixXd massMatrix(const moveit::core::RobotState& state) const;

  /// Leading blocks of the mass matrix.
  void coupling(const moveit::core::RobotState& state,
                Eigen::Matrix<double, 6, 6>& base_inertia,
                Eigen::MatrixXd& coupling_block) const;

  /// J_m, 6 x n, the fixed base end effector Jacobian, [linear; angular].
  Eigen::MatrixXd manipulatorJacobian(const moveit::core::RobotState& state) const;

  /// J_b, 6 x 6, end effector twist produced by a base twist.
  Eigen::Matrix<double, 6, 6> baseJacobian(const moveit::core::RobotState& state) const;

  /// J_g = J_m - J_b H_b^-1 H_bm, 6 x n.
  Eigen::MatrixXd generalizedJacobian(const moveit::core::RobotState& state) const;

  /// v_b = -H_b^-1 H_bm qdot, [linear, angular] about the reference point.
  Eigen::Matrix<double, 6, 1> baseVelocity(const moveit::core::RobotState& state,
                                           const Eigen::VectorXd& joint_rates) const;

private:
  /// 6 x 6 mapping a base twist at \p reference to the twist at \p point.
  /// Shared by the mass matrix and by J_b so their conventions cannot drift.
  Eigen::Matrix<double, 6, 6> twistTransport(const Eigen::Vector3d& point,
                                             const Eigen::Vector3d& reference) const;

  moveit::core::RobotModelConstPtr model_;
  std::vector<std::string> arm_joints_;
  std::vector<const moveit::core::LinkModel*> arm_child_links_;
  std::vector<Eigen::Vector3d> arm_axes_;
  std::vector<LinkInertia> links_;
  const moveit::core::LinkModel* end_effector_ = nullptr;
  std::vector<std::size_t> end_effector_ancestors_;
  const moveit::core::LinkModel* base_link_ = nullptr;
  Eigen::Isometry3d base_inertial_transform_ = Eigen::Isometry3d::Identity();
};

}  // namespace free_floating_manipulation

#endif  // FREE_FLOATING_MANIPULATION_DYNAMICS_HPP
