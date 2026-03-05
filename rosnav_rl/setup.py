from setuptools import setup, find_packages
import os
from glob import glob

package_name = "rosnav_rl"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Arena-Rosnav",
    maintainer_email="arena-rosnav@users.noreply.github.com",
    description="DRL-based ROS2 navigation controller with action server",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "action_server = scripts.action_server:main",
        ],
    },
)
