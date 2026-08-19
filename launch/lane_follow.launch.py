#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import os


def generate_launch_description():
    package_share = get_package_share_directory('physicar_autonomy')
    default_params = os.path.join(package_share, 'config', 'lane_follow.yaml')
    default_lidar_params = os.path.join(package_share, 'config', 'lidar_obstacle.yaml')
    rviz_config = os.path.join(package_share, 'rviz', 'lane_follow.rviz')

    params_file = LaunchConfiguration('params_file')
    use_rviz = LaunchConfiguration('use_rviz')
    control_enabled = LaunchConfiguration('control_enabled')
    use_lidar = LaunchConfiguration('use_lidar')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the PhysiCar lane-following parameter file',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Start RViz with the lane debug image display',
        ),
        DeclareLaunchArgument(
            'control_enabled',
            default_value='false',
            description='Publish Pure Pursuit commands to /speed and /steering',
        ),
        DeclareLaunchArgument(
            'use_lidar',
            default_value='true',
            description='Start front LiDAR clustering and RViz markers',
        ),
        Node(
            package='physicar_autonomy',
            executable='lane_follow_node',
            name='lane_follow',
            output='screen',
            parameters=[
                params_file,
                {'control_enabled': ParameterValue(control_enabled, value_type=bool)},
            ],
        ),
        Node(
            package='physicar_autonomy',
            executable='lidar_obstacle_node',
            name='lidar_obstacle',
            output='screen',
            parameters=[default_lidar_params],
            condition=IfCondition(use_lidar),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='lane_debug_rviz',
            output='screen',
            arguments=['-d', rviz_config],
            condition=IfCondition(use_rviz),
        ),
    ])
