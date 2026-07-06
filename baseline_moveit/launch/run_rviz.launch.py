import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            # Nếu bạn có file .rviz cấu hình sẵn thì thêm dòng dưới vào, 
            # còn không thì nó sẽ mở RViz trắng để bạn tự Add PlanningScene
            # arguments=['-d', os.path.join(get_package_share_directory('my_robot_control'), 'config', 'my_config.rviz')]
        )
    ])
