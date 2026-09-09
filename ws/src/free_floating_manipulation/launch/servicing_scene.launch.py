"""Bring up the servicing scene, pinned to one Gazebo.

There are two Gazebo Sim installations on this machine and they are not the
same version. The standalone `gz-sim8-cli` is 8.15.0; the one ROS 2 pulls in
through `ros-jazzy-gz-sim-vendor` is 8.11.0. Which one runs is decided by
`GZ_CONFIG_PATH` and not by `PATH`: measured, sourcing ROS gives 8.11.0 even
when `/usr/bin` is first on `PATH` and even when `/usr/bin/gz` is invoked by
absolute path, because that binary is a launcher that loads whatever the
config path points at.

This launch file therefore sets `GZ_CONFIG_PATH` explicitly rather than
inheriting it, and logs the version it resolved. The vendored 8.11.0 is the
deliberate choice, because `ros_gz_bridge` was built against it and a scene
whose renderer and bridge disagree about their own libraries is a bad place
to spend a session.
"""

import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

VENDOR = "/opt/ros/jazzy/opt"
VENDOR_GZ_CONFIG = ":".join(
    f"{VENDOR}/{p}/share/gz" for p in
    ("gz_sim_vendor", "sdformat_vendor", "gz_gui_vendor", "gz_transport_vendor",
     "gz_rendering_vendor", "gz_plugin_vendor", "gz_fuel_tools_vendor",
     "gz_msgs_vendor", "gz_common_vendor"))


def resolved_version(config_path):
    env = dict(os.environ, GZ_CONFIG_PATH=config_path)
    try:
        out = subprocess.run(["gz", "sim", "--version"], env=env,
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            if "version" in line:
                return line.strip()
    except Exception as exc:
        return f"could not be determined ({exc})"
    return "unknown"


def generate_launch_description():
    share = get_package_share_directory("free_floating_manipulation")
    world = os.path.join(share, "models", "servicing_scene.sdf")

    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="true"),
        DeclareLaunchArgument(
            "gz_config_path", default_value=VENDOR_GZ_CONFIG,
            description="Pins which Gazebo runs. PATH does not decide this."),
        LogInfo(msg=["Gazebo pinned by GZ_CONFIG_PATH resolves to: ",
                     resolved_version(VENDOR_GZ_CONFIG)]),
        ExecuteProcess(
            cmd=["gz", "sim", "-s", "-r", "-v", "2", world],
            output="screen",
            additional_env={"GZ_CONFIG_PATH": LaunchConfiguration("gz_config_path")}),
        Node(
            package="free_floating_manipulation",
            executable="scene_publisher.py",
            name="servicing_scene",
            output="screen",
            parameters=[{"gazebo": True}]),
    ])
