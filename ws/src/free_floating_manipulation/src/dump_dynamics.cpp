// Dump the C++ dynamics over a set of configurations, for comparison against
// the Python implementation.
//
// Nothing here checks the numbers. It writes them, and
// examples/verify_cpp_dynamics.py decides whether they agree, per block and
// per link, so that a disagreement says where it is rather than only that it
// exists. That is the same discipline that localised the ancestor masking bug
// in phase 2, where an aggregate norm would have said 0.3 and pointed nowhere.
//
// Usage:
//   dump_dynamics <urdf path> <configurations file> <output file>
//
// The configurations file holds one configuration per line, seven joint
// angles, whitespace separated.

#include "free_floating_manipulation/dynamics.hpp"

#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

const std::vector<std::string> kArmJoints = { "panda_joint1", "panda_joint2", "panda_joint3",
                                              "panda_joint4", "panda_joint5", "panda_joint6",
                                              "panda_joint7" };


void writeMatrix(std::ostream& out, const std::string& name, const Eigen::MatrixXd& m)
{
  out << name << ' ' << m.rows() << ' ' << m.cols() << '\n';
  for (Eigen::Index r = 0; r < m.rows(); ++r)
  {
    for (Eigen::Index c = 0; c < m.cols(); ++c)
      out << (c ? " " : "") << m(r, c);
    out << '\n';
  }
}

std::string readFile(const std::string& path)
{
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("cannot open " + path);
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 5)
  {
    std::cerr << "usage: dump_dynamics <urdf> <configurations> <output> <end effector link>\n";
    return 2;
  }
  const std::string end_effector = argv[4];

  const std::string urdf_path = argv[1];
  urdf::ModelInterfaceSharedPtr urdf_model = urdf::parseURDF(readFile(urdf_path));
  if (!urdf_model)
  {
    std::cerr << "failed to parse URDF: " << urdf_path << '\n';
    return 1;
  }

  // A RobotModel needs an SRDF. Nothing here uses planning groups, so the
  // minimum that satisfies the constructor is the right amount.
  auto srdf_model = std::make_shared<srdf::Model>();
  if (!srdf_model->initString(*urdf_model, "<robot name=\"panda\"></robot>"))
  {
    std::cerr << "failed to initialise a minimal SRDF\n";
    return 1;
  }

  auto robot_model = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
  free_floating_manipulation::FloatingBaseModel model(robot_model, kArmJoints, end_effector);
  moveit::core::RobotState state(robot_model);
  state.setToDefaultValues();

  std::vector<std::vector<double>> configurations;
  {
    std::ifstream stream(argv[2]);
    std::string line;
    while (std::getline(stream, line))
    {
      std::istringstream parts(line);
      std::vector<double> angles;
      double value = 0.0;
      while (parts >> value)
        angles.push_back(value);
      if (angles.size() == kArmJoints.size())
        configurations.push_back(angles);
    }
  }

  std::ofstream out(argv[3]);
  out << std::setprecision(17);
  out << "MODEL_FRAME " << robot_model->getModelFrame() << '\n';
  out << "TOTAL_MASS " << model.totalMass() << '\n';
  out << "NCONFIG " << configurations.size() << '\n';

  // Per link constants, so a disagreement in the model can be told apart from
  // a disagreement in the arithmetic built on top of it.
  for (const auto& link : model.links())
  {
    out << "LINK " << link.name << ' ' << link.mass << ' '
        << link.inertial_transform.translation().transpose() << ' ';
    for (Eigen::Index r = 0; r < 3; ++r)
      for (Eigen::Index c = 0; c < 3; ++c)
        out << link.inertia(r, c) << ' ';
    out << "ancestors";
    for (std::size_t a : link.arm_ancestors)
      out << ' ' << a;
    out << '\n';
  }

  for (std::size_t i = 0; i < configurations.size(); ++i)
  {
    for (std::size_t j = 0; j < kArmJoints.size(); ++j)
      state.setJointPositions(kArmJoints[j], &configurations[i][j]);
    state.update();

    out << "CONFIG " << i << '\n';

    // Frames first. If these disagree nothing downstream can agree.
    for (const auto& link : model.links())
    {
      const Eigen::Isometry3d link_frame = state.getGlobalLinkTransform(link.link);
      out << "LINKFRAME " << link.name << ' ' << link_frame.translation().transpose() << '\n';
      out << "COM " << link.name << ' '
          << model.comPose(link, state).translation().transpose() << '\n';
    }

    const auto frames = model.armJointFrames(state);
    for (std::size_t j = 0; j < frames.size(); ++j)
      out << "JOINTFRAME " << j << ' ' << frames[j].point.transpose() << ' '
          << frames[j].axis.transpose() << '\n';

    // Per link Jacobians, which is where the ancestor masking bug lived.
    const Eigen::Index n = static_cast<Eigen::Index>(model.dofs());
    Eigen::MatrixXd linear(3, n), angular(3, n);
    for (const auto& link : model.links())
    {
      model.geometricColumns(model.comPose(link, state).translation(), frames,
                             link.arm_ancestors, linear, angular);
      writeMatrix(out, "JT " + link.name, linear);
      writeMatrix(out, "JR " + link.name, angular);
    }

    Eigen::Matrix<double, 6, 6> base_inertia;
    Eigen::MatrixXd coupling_block;
    model.coupling(state, base_inertia, coupling_block);

    writeMatrix(out, "M", model.massMatrix(state));
    writeMatrix(out, "HB", base_inertia);
    writeMatrix(out, "HBM", coupling_block);
    writeMatrix(out, "JM", model.manipulatorJacobian(state));
    writeMatrix(out, "JB", model.baseJacobian(state));
    writeMatrix(out, "JG", model.generalizedJacobian(state));
  }

  std::cout << "wrote " << configurations.size() << " configurations to " << argv[3] << '\n';
  return 0;
}
