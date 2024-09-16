#!/bin/sh

# Find the path to the on_orbit package
PACKAGE_PATH=$(rospack find on_orbit)

echo -n "Creating robot_cv_detached.urdf"
xacro $PACKAGE_PATH/urdf/simulation/satellite/robot_cv_detached.xacro -o robot_cv_detached.urdf

echo -n "Creating robot.urdf"
xacro $PACKAGE_PATH/urdf/simulation/satellite/robot.xacro -o robot.urdf

echo -n "Creating cv.urdf"
xacro $PACKAGE_PATH/urdf/simulation/satellite/cv.xacro -o cv.urdf

echo -n "Creating on_orbit_rail.urdf"
xacro $PACKAGE_PATH/urdf/hardware/on_orbit.xacro -o on_orbit_rail.urdf

echo -n "Creating holodeck.urdf"
xacro $PACKAGE_PATH/urdf/hardware/holodeck.xacro -o holodeck.urdf

echo "Done."