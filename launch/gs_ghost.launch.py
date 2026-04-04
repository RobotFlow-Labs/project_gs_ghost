"""ROS2 launch file for GS-GHOST node."""

from __future__ import annotations


def generate_launch_description():
    try:
        from launch import LaunchDescription
        from launch_ros.actions import Node

        return LaunchDescription([
            Node(
                package="anima_gs_ghost",
                executable="gs_ghost_node",
                name="gs_ghost_node",
                output="screen",
                parameters=[{
                    "device": "cuda:1",
                    "sfm_method": "vggsfm",
                }],
            ),
        ])
    except ImportError:
        print("launch_ros not available — install ros2 packages first")
        return None
