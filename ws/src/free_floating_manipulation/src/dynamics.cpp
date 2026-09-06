#include "free_floating_manipulation/dynamics.hpp"

#include <moveit/robot_model/joint_model.hpp>
#include <moveit/robot_model/link_model.hpp>
#include <moveit/robot_model/revolute_joint_model.hpp>

#include <algorithm>
#include <stdexcept>
#include <urdf_model/link.h>

namespace free_floating_manipulation
{
namespace
{

/// URDF inertial origin as an Eigen transform.
Eigen::Isometry3d inertialTransform(const urdf::Inertial& inertial)
{
  const urdf::Pose& pose = inertial.origin;
  Eigen::Quaterniond rotation(pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z);
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = rotation.normalized().toRotationMatrix();
  transform.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  return transform;
}

/// The full symmetric inertia tensor, about the centre of mass, in the
/// inertial frame. The off diagonal terms are read rather than assumed zero:
/// the Panda URDF PyBullet ships happens to be diagonal, other Panda models
/// are not, and silently dropping them would be a real modelling error.
Eigen::Matrix3d inertiaTensor(const urdf::Inertial& inertial)
{
  Eigen::Matrix3d inertia;
  inertia << inertial.ixx, inertial.ixy, inertial.ixz,
             inertial.ixy, inertial.iyy, inertial.iyz,
             inertial.ixz, inertial.iyz, inertial.izz;
  return inertia;
}

}  // namespace

Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
  Eigen::Matrix3d s;
  s <<   0.0, -v.z(),  v.y(),
       v.z(),    0.0, -v.x(),
      -v.y(),  v.x(),    0.0;
  return s;
}

FloatingBaseModel::FloatingBaseModel(const moveit::core::RobotModelConstPtr& model,
                                     const std::vector<std::string>& arm_joints,
                                     const std::string& end_effector_link,
                                     double base_mass,
                                     const Eigen::Vector3d& base_inertia_diagonal)
  : model_(model), arm_joints_(arm_joints)
{
  const auto& urdf_model = model_->getURDF();
  if (!urdf_model)
    throw std::runtime_error("the RobotModel carries no URDF, so there is no inertial data to read");

  // Arm joints must be revolute: the geometric Jacobian columns below are the
  // revolute form and would be silently wrong for a prismatic joint.
  for (const std::string& name : arm_joints_)
  {
    const moveit::core::JointModel* joint = model_->getJointModel(name);
    if (joint == nullptr)
      throw std::runtime_error("no such joint: " + name);
    if (joint->getType() != moveit::core::JointModel::REVOLUTE)
      throw std::runtime_error("arm joint " + name + " is not revolute");

    const auto* revolute = static_cast<const moveit::core::RevoluteJointModel*>(joint);
    arm_axes_.push_back(revolute->getAxis());
    arm_child_links_.push_back(joint->getChildLinkModel());
  }

  base_link_ = model_->getRootLink();
  end_effector_ = model_->getLinkModel(end_effector_link);
  if (end_effector_ == nullptr)
    throw std::runtime_error("no such link: " + end_effector_link);

  // Which arm joints lie on the chain to a given link. Walking up the parent
  // chain is the only way to know; the geometric formula cannot tell.
  auto ancestors_of = [this](const moveit::core::LinkModel* link) {
    std::vector<std::size_t> found;
    for (const moveit::core::LinkModel* walk = link; walk != nullptr;
         walk = walk->getParentLinkModel())
    {
      const moveit::core::JointModel* joint = walk->getParentJointModel();
      if (joint == nullptr)
        break;
      auto it = std::find(arm_joints_.begin(), arm_joints_.end(), joint->getName());
      if (it != arm_joints_.end())
        found.push_back(static_cast<std::size_t>(std::distance(arm_joints_.begin(), it)));
    }
    std::sort(found.begin(), found.end());
    return found;
  };

  for (const moveit::core::LinkModel* link : model_->getLinkModels())
  {
    urdf::LinkConstSharedPtr urdf_link = urdf_model->getLink(link->getName());
    LinkInertia entry;
    entry.link = link;
    entry.name = link->getName();
    if (urdf_link && urdf_link->inertial)
    {
      entry.mass = urdf_link->inertial->mass;
      entry.inertial_transform = inertialTransform(*urdf_link->inertial);
      entry.inertia = inertiaTensor(*urdf_link->inertial);
    }
    entry.arm_ancestors = ancestors_of(link);

    if (link == base_link_)
    {
      if (base_mass > 0.0)
        entry.mass = base_mass;
      if ((base_inertia_diagonal.array() > 0.0).all())
        entry.inertia = base_inertia_diagonal.asDiagonal();
      base_inertial_transform_ = entry.inertial_transform;
    }
    links_.push_back(entry);
  }

  end_effector_ancestors_ = ancestors_of(end_effector_);
}

double FloatingBaseModel::totalMass() const
{
  double total = 0.0;
  for (const LinkInertia& link : links_)
    total += link.mass;
  return total;
}

std::vector<JointFrame> FloatingBaseModel::armJointFrames(const moveit::core::RobotState& state) const
{
  std::vector<JointFrame> frames(arm_joints_.size());
  for (std::size_t i = 0; i < arm_joints_.size(); ++i)
  {
    // The child link frame coincides with the joint frame, and rotating about
    // the joint axis leaves the axis itself unchanged, so the child link's
    // global rotation carries the axis into the world correctly.
    const Eigen::Isometry3d& transform = state.getGlobalLinkTransform(arm_child_links_[i]);
    frames[i].point = transform.translation();
    frames[i].axis = transform.linear() * arm_axes_[i];
  }
  return frames;
}

void FloatingBaseModel::geometricColumns(const Eigen::Vector3d& target,
                                         const std::vector<JointFrame>& frames,
                                         const std::vector<std::size_t>& ancestors,
                                         Eigen::Ref<Eigen::MatrixXd> linear,
                                         Eigen::Ref<Eigen::MatrixXd> angular) const
{
  linear.setZero();
  angular.setZero();
  for (std::size_t index : ancestors)
  {
    const JointFrame& frame = frames[index];
    linear.col(static_cast<Eigen::Index>(index)) = frame.axis.cross(target - frame.point);
    angular.col(static_cast<Eigen::Index>(index)) = frame.axis;
  }
}

Eigen::Isometry3d FloatingBaseModel::comPose(const LinkInertia& link,
                                             const moveit::core::RobotState& state) const
{
  return state.getGlobalLinkTransform(link.link) * link.inertial_transform;
}

Eigen::Vector3d FloatingBaseModel::referencePoint(const moveit::core::RobotState& state) const
{
  return (state.getGlobalLinkTransform(base_link_) * base_inertial_transform_).translation();
}

Eigen::Matrix<double, 6, 6> FloatingBaseModel::twistTransport(const Eigen::Vector3d& point,
                                                              const Eigen::Vector3d& reference) const
{
  Eigen::Matrix<double, 6, 6> transport = Eigen::Matrix<double, 6, 6>::Identity();
  transport.topRightCorner<3, 3>() = -skew(point - reference);
  return transport;
}

Eigen::MatrixXd FloatingBaseModel::massMatrix(const moveit::core::RobotState& state) const
{
  const Eigen::Index n = static_cast<Eigen::Index>(dofs());
  const Eigen::Index size = 6 + n;
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(size, size);

  const Eigen::Vector3d reference = referencePoint(state);
  const std::vector<JointFrame> frames = armJointFrames(state);

  Eigen::MatrixXd linear_arm(3, n);
  Eigen::MatrixXd angular_arm(3, n);
  Eigen::MatrixXd jacobian(6, size);

  for (const LinkInertia& link : links_)
  {
    if (link.mass == 0.0)
      continue;

    const Eigen::Isometry3d com = comPose(link, state);
    geometricColumns(com.translation(), frames, link.arm_ancestors, linear_arm, angular_arm);

    jacobian.leftCols<6>() = twistTransport(com.translation(), reference);
    jacobian.topRightCorner(3, n) = linear_arm;
    jacobian.bottomRightCorner(3, n) = angular_arm;

    const Eigen::Matrix3d rotation = com.linear();
    const Eigen::Matrix3d inertia_world = rotation * link.inertia * rotation.transpose();

    matrix.noalias() += link.mass * jacobian.topRows<3>().transpose() * jacobian.topRows<3>();
    matrix.noalias() += jacobian.bottomRows<3>().transpose() * inertia_world * jacobian.bottomRows<3>();
  }
  return matrix;
}

void FloatingBaseModel::coupling(const moveit::core::RobotState& state,
                                 Eigen::Matrix<double, 6, 6>& base_inertia,
                                 Eigen::MatrixXd& coupling_block) const
{
  const Eigen::MatrixXd matrix = massMatrix(state);
  base_inertia = matrix.topLeftCorner<6, 6>();
  coupling_block = matrix.topRightCorner(6, static_cast<Eigen::Index>(dofs()));
}

Eigen::MatrixXd FloatingBaseModel::manipulatorJacobian(const moveit::core::RobotState& state) const
{
  const Eigen::Index n = static_cast<Eigen::Index>(dofs());
  Eigen::MatrixXd linear(3, n);
  Eigen::MatrixXd angular(3, n);
  const std::vector<JointFrame> frames = armJointFrames(state);
  const Eigen::Vector3d target = state.getGlobalLinkTransform(end_effector_).translation();
  geometricColumns(target, frames, end_effector_ancestors_, linear, angular);

  Eigen::MatrixXd jacobian(6, n);
  jacobian.topRows<3>() = linear;
  jacobian.bottomRows<3>() = angular;
  return jacobian;
}

Eigen::Matrix<double, 6, 6> FloatingBaseModel::baseJacobian(const moveit::core::RobotState& state) const
{
  return twistTransport(state.getGlobalLinkTransform(end_effector_).translation(),
                        referencePoint(state));
}

Eigen::MatrixXd FloatingBaseModel::generalizedJacobian(const moveit::core::RobotState& state) const
{
  Eigen::Matrix<double, 6, 6> base_inertia;
  Eigen::MatrixXd coupling_block;
  coupling(state, base_inertia, coupling_block);

  // Same guard the Python carries: in [linear, angular] order the leading
  // block of H_b is the translational inertia and must be total mass times
  // identity. In [angular, linear] order it would be the rotational inertia,
  // and J_g would come out the right shape and wrong everywhere.
  const double leading = (base_inertia.topLeftCorner<3, 3>()
                          - totalMass() * Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
  if (leading > 1e-6)
    throw std::runtime_error("H_b leading block is not total mass times identity; "
                             "the base degrees of freedom are not [linear, angular]");

  return manipulatorJacobian(state)
         - baseJacobian(state) * base_inertia.ldlt().solve(coupling_block);
}

Eigen::Matrix<double, 6, 1> FloatingBaseModel::baseVelocity(const moveit::core::RobotState& state,
                                                            const Eigen::VectorXd& joint_rates) const
{
  Eigen::Matrix<double, 6, 6> base_inertia;
  Eigen::MatrixXd coupling_block;
  coupling(state, base_inertia, coupling_block);
  return -base_inertia.ldlt().solve(coupling_block * joint_rates);
}

}  // namespace free_floating_manipulation
