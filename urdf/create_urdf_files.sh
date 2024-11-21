#!/bin/sh

# Find the path to the on_orbit package
PACKAGE_PATH=$(rospack find on_orbit)

echo -n "Creating robot_cv_detached.urdf"
xacro $PACKAGE_PATH/urdf/simulation/satellite/robot_cv_detached.xacro -o robot_cv_detached.urdf

echo -n "Create single_arm.urdf"
xacro $PACKAGE_PATH/urdf/hardware/ur/ur_1_ur10e.xacro -o single_arm.urdf

echo -n "Creating robot.urdf"
xacro $PACKAGE_PATH/urdf/simulation/satellite/robot.xacro -o robot.urdf

echo -n "Creating cv.urdf"
xacro $PACKAGE_PATH/urdf/simulation/satellite/cv.xacro -o cv.urdf

echo -n "Creating on_orbit.urdf"
xacro $PACKAGE_PATH/urdf/hardware/on_orbit.xacro -o on_orbit.urdf

echo -n "Creating holodeck.urdf"
xacro $PACKAGE_PATH/urdf/hardware/holodeck.xacro -o holodeck.urdf

echo -n "Creating mrv_ur_arm.urdf"
xacro $PACKAGE_PATH/urdf/hardware/ur/ur_3_ur10e.xacro -o mrv_ur_arm.urdf

echo "Done."