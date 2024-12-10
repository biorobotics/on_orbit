# User Guide for NG Space Project HIL System

**Last Updated:** December 10, 2024

This guide provides instructions for setting up, building, and running the NG Space Project Hardware-In-The-Loop (HIL) system. It covers simulation environments, controllers, and trajectory optimization.

---

## Table of Contents
- [User Guide for NG Space Project HIL System](#user-guide-for-ng-space-project-hil-system)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Repositories](#repositories)
  - [System Requirements](#system-requirements)
  - [Dependencies \& Installation](#dependencies--installation)
  - [Building the Code](#building-the-code)
  - [Running the System](#running-the-system)
  - [Trajectory Optimization \& Virtual Environments](#trajectory-optimization--virtual-environments)
  - [Replaying Trials \& Logging](#replaying-trials--logging)
  - [Plotting \& FLOP Counting](#plotting--flop-counting)
  - [Additional Notes \& Tips](#additional-notes--tips)
  - [Code Structure \& URDFs](#code-structure--urdfs)

---

## Overview
The **NG Space Project HIL System** enables both simulations and hardware-in-the-loop experiments for on-orbit manipulation tasks. It integrates simulation environments (e.g., **Pinocchio**, **MuJoCo**) with robotics frameworks (e.g., **ROS**, **UR arms**, **Vention hardware**) and advanced control methodologies (e.g., **Crocoddyl**, trajectory optimization, MPC, TVLQR).

You will create a `catkin_ws` workspace and clone the relevant repositories into its `src` directory. This system supports:
- Trajectory generation
- Model Predictive Control (MPC)
- Time-Varying Linear Quadratic Regulator (TVLQR)
- Pure simulation and hardware-in-the-loop runs

---

## Repositories

1. **on_orbit (Active Repository)**  
   - **URL:** [https://github.com/biorobotics/on_orbit](https://github.com/biorobotics/on_orbit)  
   - **Branch:** `main`  
   - Primary repository with up-to-date code and instructions.

2. **peg_in_hole (Legacy Repository)**  
   - **URL:** [https://github.com/SURI-Shared/peg_in_hole.git](https://github.com/SURI-Shared/peg_in_hole.git)  
   - **Branch:** `master`  
   - Contains older hardware experiment code. Check here if something is missing in `on_orbit`.

3. **space_robot (Reference Only)**  
   - **URL:** [https://github.com/biorobotics/space_robot.git](https://github.com/biorobotics/space_robot.git)  
   - Not actively used. Provides an overview of related packages.

---

## System Requirements
- **Operating System:** Ubuntu 20.04 (Focal Fossa) recommended for ROS Noetic compatibility.
- **ROS Distribution:** ROS Noetic recommended. (Melodic may work but is not preferred.)
- **Virtual Environments:** Do **not** use conda environments. Use a standard Python virtual environment and/or `pip`.

---

## Dependencies & Installation

**Important:** Do **not** use conda. Create a Python virtual environment or install packages system-wide as needed.

**Required Packages:**

- **ROS Noetic**  
  Installation guide: [http://wiki.ros.org/noetic/Installation/Ubuntu](http://wiki.ros.org/noetic/Installation/Ubuntu)

- **Pinocchio**  
  Installation guide: [https://stack-of-tasks.github.io/pinocchio/download.html](https://stack-of-tasks.github.io/pinocchio/download.html)  
  **Note:** Do **not** install `ros-noetic-pinocchio` via apt. Build from source.

- **Crocoddyl**  
  Installation guide: [https://github.com/loco-3d/crocoddyl](https://github.com/loco-3d/crocoddyl)
  
- **OSQP**  
  [https://osqp.org/docs/get_started/sources.html](https://osqp.org/docs/get_started/sources.html)  
  Use **release v0.6.3**:
  ```bash
  git clone --recursive https://github.com/osqp/osqp
  cd osqp
  git checkout v0.6.3
  git submodule update --recursive

- **OSQP-Eigen**  
  [https://github.com/robotology/osqp-eigen](https://github.com/robotology/osqp-eigen)  
  Build from source (e.g., v0.7.0).

- **ifopt**  
  Install via ROS binaries:
  sudo apt-get install ros-noetic-ifopt

- **CppAD & CppADCodeGen**  
  [https://github.com/coin-or/CppAD](https://github.com/coin-or/CppAD) 
  [https://github.com/joaoleal/CppADCodeGen](https://github.com/joaoleal/CppADCodeGen)  
  Build from source.

- **ViSP**  
  [https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-install-ubuntu.html](https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-install-ubuntu.html)

- **pybind11**  
  pip install pybind11  
  Add export PATH=~/.local/bin:$PATH to ~/.bashrc if needed.

- **MuJoCo (v2.1.0)**  
  [https://mujoco.org/download](https://mujoco.org/download)  
  Extract `mujoco210` into `~/.mujoco/mujoco210`.  

  Add the following lines to `~/.bashrc`:
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
  ```

  Then:
```bash
  pip install "cython<3"
  pip install mujoco_py
  ```

  In Python:
```python
  import mujoco_py
  ```


- **scikit-sparse** 
```bash 
  sudo apt-get install libsuitesparse-dev  
  pip install scikit-sparse  
  pip install -U scipy
  ```

- **Mosek** (optional)  
  Download Mosek and set up path and license. Build Mosek Fusion C++ API from source if needed.

- **Casadi** 
  Build from source.
```bash  
  sudo apt install swig
  ``` 
  may be required.

- **Experimental Pinocchio (JNRH-2023) with CasADi Support**  
  If you need trajectory generation using CasADi-based Pinocchio calls, build this special version.  
  Use Python 3.8 venv.  
  Consider newer Pinocchio versions to avoid this step.

**Recommended Virtual Environment Setup:**
```bash
  python3.8 -m venv ~/.venvs/my-venv-name  
  source ~/.venvs/my-venv-name/bin/activate  
  pip install --upgrade pip setuptools wheel  
  pip install "cython<3"  
  pip install cyipopt==1.2.0  
  pip install -r requirements.txt
  ```

This venv is needed if you run trajectory generation, MPC, or TVLQR that depend on CasADi within Pinocchio.

## Building the Code

1. Catkin Workspace Setup:
```bash
   mkdir -p ~/catkin_ws/src  
   cd ~/catkin_ws/src  
   git clone https://github.com/biorobotics/on_orbit.git  
   cd ~/catkin_ws
   ```

2. Building:
   Use catkin build (not catkin_make):
```bash
   catkin build on_orbit -DCMAKE_BUILD_TYPE=Release -j2
  ```

   For debugging:
```bash
   catkin build on_orbit -DCMAKE_BUILD_TYPE=Debug -j2
  ```
   Debug mode: safer for development.  
   Release mode: required for real-time hardware performance.

3. Pybind11 path (if needed):
```bash
   catkin build on_orbit -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=~/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11/ -j2
  ```

4. Eigen Issues:
   If eigen is not found:
```bash
  sudo ln -sf /usr/include/eigen3/Eigen /usr/local/include/Eigen  
  sudo ln -sf /usr/include/eigen3/unsupported /usr/local/include/unsupported
  ```

5. Switching Between Debug/Release:
   If build mode doesn’t change properly:
```bash
     catkin clean  
     catkin build on_orbit -DCMAKE_BUILD_TYPE=Release -j2
```
---

## Running the System

Hardware-In-The-Loop (HIL):
  Launch netft nodes:
```bash  
    roslaunch netft_utils netft_single.launch
```  
  Launch UR Holodeck nodes:
```bash  
    roslaunch ur_state_machine ur_state_machine.launch
```  
  Launch Vention Holodeck nodes:
```bash  
    roslaunch vention_control vention_node.launch
```

Then run:
```bash
  roslaunch on_orbit hw_controller.launch
  ```

Follow on-screen instructions.

---

## Trajectory Optimization & Virtual Environments

For generating trajectories:
1. Use the .venv with experimental Pinocchio if needed.
2. cd `on_orbit/scripts/planning_scripts/ipopt_cpp/`
```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make
   ```
3. source your .venv
```bash
   cd ..
   PYTHONPATH=. python3 run_ipopt_trajopt.py
   For dense library:
   PYTHONPATH=. python3 run_ipopt_trajopt_alt.py
   ```

If counting FLOPs:
```bash
  sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
  ```
Or set count_total_flop=False to skip FLOP counting.

---

## Replaying Trials & Logging

To replay simulations:
```bash
  roslaunch on_orbit replay_sim.launch
  ```
Set load_path and replay_from_xs in `replay_sim_node.py`.

For hardware trials replay (WIP):
```bash
  roslaunch on_orbit replay_hil_and_sim.launch
  ```

To replay hardware joint angles on hardware:
  Use `replay_two_arm_hw_experiment_on_hw.py` and associated launch file.

Data logging directories specified in config/experiment.yaml.

---

## Plotting & FLOP Counting

Use plotting scripts in utility/ directory.
Edit Python scripts for FLOP counting if needed.

---

## Additional Notes & Tips

- MPC & TVLQR currently only tested in pure simulation.
- For trajectory generation, MPC, TVLQR:
  Activate venv and run `simulator_node_MPC.py` or `simulator_node_TVLQR.py` in hil_sim_scripts.
- Gravity compensation or sensor calibration:
```bash
  python3 ft_calibration.py
  ```
  Ensure safe configuration of arms first.

---

## Code Structure & URDFs

- **Python code:**
  mrv_client_sim.py (main simulation)
  hil_runner.py (HIL)
  ipopt_contact_planner.py, ipopt_contact_script.py (trajectory optimization)

- **C++ code:**
  In src folder of on_orbit, includes controllers and bindings.

- **URDFs:**
  In urdf/.
  If peg length changes, edit ur_3_ur10e.xacro or ur_4_ur10e.xacro.
  If nozzle/peg geometry changes, edit cv.xacro, robot.urdf, robot_cv_detached.xacro.
  Run ./create_urdf_files.sh in urdf/ to regenerate URDFs.

- **Media & Data:**
  For video compression:
    for i in *.mp4; do ffmpeg -i "$i" "${i%.*}_c.mp4"; done

---
