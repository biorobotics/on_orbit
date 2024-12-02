# On-Orbit
Simulation for On-Orbit Manipulation

##  Contents
* **[Installation](#installation)**
* **[Executing](#Executing)**

## Installation

### Ubuntu enviroment
#### Dependencies
- [ROS](#ROS)
- [pinocchio](#pinocchio)
- [crocoddyl](#crocoddyl)
- [ifopt](#ifopt)
- [CppAD](#CppAD)
- [CppADCodeGen](#CppADCodeGen)
- [pybind11](#pybind11)
- [OSQP](#OSQP)
- [osqp-eigen](#osqp-eigen)
- [scikit-sparse](#scikit-sparse)

#### [ROS](https://www.ros.org/)
This code has been tested in ROS Noetic on Ubuntu 20.04. See the installation instructions for [Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu).

#### [pinocchio](https://github.com/stack-of-tasks/pinocchio)
Follow the installation instructions [here](https://stack-of-tasks.github.io/pinocchio/download.html). Note, please do NOT install ros-noetic-pinocchio using apt. For whatever reason, the code doesn't work with that. Use the instructions given in the link to pinocchio's website.

#### [crocoddyl](https://github.com/loco-3d/crocoddyl)
Follow the installation instructions on the GitHub page, linked above.

#### [OSQP](https://osqp.org/)
Follow the installation instructions [here](https://osqp.org/docs/get_started/sources.html).

#### [osqp-eigen](https://github.com/robotology/osqp-eigen)
Follow the installation instructions [here](https://github.com/robotology/osqp-eigen).

#### [ifopt](https://github.com/ethz-adrl/ifopt)
Follow the installation instructions on the GitHub page, linked above.

#### [CppAD](https://github.com/coin-or/CppAD)
Download the source code and run the usual cmake commands from the repository's root directory:
```cmd
mkdir build
cmake ..
make
sudo make install
```

#### [CppADCodeGen](https://github.com/joaoleal/CppADCodeGen)
Download the source code and run the usual cmake commands from the repository's root directory.

#### [ViSP](https://github.com/lagadic/visp)
Install the prereqs and recommended 3rd partly libraries mentioned [here](https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-install-ubuntu.html), then download the source code and run the usual cmake commands from the repository's root directory. On the website, they describe creating a workspace, but you don't need to do this.

#### [pybind11](https://pybind11.readthedocs.io/en/stable/)
Install with pip, as detailed [here](https://pybind11.readthedocs.io/en/stable/installing.html). You might need to add

```cmd
$ export PATH=~/.local/bin:$PATH
```
to your $HOME/.bashrc.

#### [MuJoCo](https://mujoco.org/)
Download version 2.1.0 from [here](https://mujoco.org/download) and extract. The folder should be called mujoco210. Create a ~/.mujoco folder and move mujoco210 into ~/.mujoco. The resulting folder should be ~/.mujoco/mujoco210. Add the following to your ~/.bashrc:
```cmd
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
also, before running any MuJoCo-related code for the first time, open up a python3 terminal and import mujoco_py, so it can do some cython-related setup.

### [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse)
Install SuiteSparse using
```cmd
$ sudo apt-get install libsuitesparse-dev
```
then install scikit-sparse using
```cmd
$ pip install scikit-sparse
```
Also, make sure your scipy is up to date, so you can use the rotation functions:
```cmd
$ pip install -U scipy
```

## Software Architecture
Whenever you test code, you'll want to run a simulator node and a controller node. Currently, the two simulator nodes available are in scripts/sim_node.py and scripts/mujoco_node.py. sim_node.py simply runs forward integration using pinocchio, while mujoco_node.py uses Mujoco, and simulates contact dynamics with the client vehicle.

Note that sim_node.py doesn't publish information about the nozzle, unlike mujoco_node.py. Therefore, in conjunction with sim_node.py, you'll want to use nozzle_node.py.

There are a couple controllers in the scripts folder, and they all extend the Controller class in controller.py. The main control node is docking_node.py. control_node.py is something I was using previously for debugging things, but it's pretty out of date.

There are various yaml files in the config folder specifying parameters for the environment and controllers.

The ROS graph is shown below:

![](rosgraph.svg)

## Executing

```cmd
$ roslaunch space_robot mujoco_docking.launch
```

To execute a basin of attraction experiment,
```cmd
$ roslaunch space_robot experiment.launch
```
and set parameters in config/experiment.yaml accordingly.

To execute a moving client experiment,
```cmd
$ roslaunch space_robot moving_client_experiment.launch
```
and set parameters in config/moving_client_experiment.yaml accordingly.

To execute a moving client experiment,
```cmd
$ roslaunch space_robot moving_client_experiment.launch
```

To run the simulation with ViSP tracking, run
```cmd
$ roslaunch space_robot mujoco_docking_visp.launch
```


##  Contents
* **[Installation](#installation)**
* **[Executing](#Executing)**

## Installation

### Ubuntu enviroment
#### Dependencies
- [On-Orbit](#on-orbit)
  - [Contents](#contents)
  - [Installation](#installation)
    - [Ubuntu enviroment](#ubuntu-enviroment)
      - [Dependencies](#dependencies)
      - [ROS](#ros)
      - [pinocchio](#pinocchio)
      - [crocoddyl](#crocoddyl)
      - [OSQP](#osqp)
      - [osqp-eigen](#osqp-eigen)
      - [ifopt](#ifopt)
      - [CppAD](#cppad)
      - [CppADCodeGen](#cppadcodegen)
      - [ViSP](#visp)
      - [pybind11](#pybind11)
      - [MuJoCo](#mujoco)
    - [scikit-sparse](#scikit-sparse)
  - [Software Architecture](#software-architecture)
  - [Executing](#executing)
  - [Contents](#contents-1)
  - [Installation](#installation-1)
    - [Ubuntu enviroment](#ubuntu-enviroment-1)
      - [Dependencies](#dependencies-1)
      - [ROS](#ros-1)
      - [pinocchio](#pinocchio-1)
      - [crocoddyl](#crocoddyl-1)
      - [OSQP](#osqp-1)
      - [osqp-eigen](#osqp-eigen-1)
      - [ifopt](#ifopt-1)
      - [CppAD](#cppad-1)
      - [CppADCodeGen](#cppadcodegen-1)
      - [ViSP](#visp-1)
      - [pybind11](#pybind11-1)
      - [MuJoCo](#mujoco-1)
    - [scikit-sparse](#scikit-sparse-1)
  - [Software Architecture](#software-architecture-1)
- [Space Robot](#space-robot)
  - [Contents](#contents-2)
  - [Installation](#installation-2)
    - [Ubuntu enviroment](#ubuntu-enviroment-2)
      - [Dependencies](#dependencies-2)
      - [ROS](#ros-2)
      - [pinocchio](#pinocchio-2)
      - [crocoddyl](#crocoddyl-2)
      - [OSQP](#osqp-2)
      - [osqp-eigen](#osqp-eigen-2)
      - [ifopt](#ifopt-2)
      - [CppAD](#cppad-2)
      - [CppADCodeGen](#cppadcodegen-2)
      - [ViSP](#visp-2)
      - [pybind11](#pybind11-2)
      - [MuJoCo](#mujoco-2)
    - [scikit-sparse](#scikit-sparse-2)
  - [Software Architecture](#software-architecture-2)
  - [Executing](#executing-1)
  - [Executing](#executing-2)

#### [ROS](https://www.ros.org/)
This code has been tested in ROS Melodic on Ubuntu 18.04, as well as ROS Noetic on Ubuntu 20.04. See the installation instructions for [Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) and [Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu).

#### [pinocchio](https://github.com/stack-of-tasks/pinocchio)
Follow the installation instructions [here](https://stack-of-tasks.github.io/pinocchio/download.html). Note, please do NOT install ros-melodic-pinocchio using apt. For whatever reason, the code doesn't work with that. Use the instructions given in the link to pinocchio's website.

#### [crocoddyl](https://github.com/loco-3d/crocoddyl)
Follow the installation instructions on the GitHub page, linked above.

#### [OSQP](https://osqp.org/)
Follow the installation instructions [here](https://osqp.org/docs/get_started/sources.html).

#### [osqp-eigen](https://github.com/robotology/osqp-eigen)
Follow the installation instructions [here](https://github.com/robotology/osqp-eigen).

#### [ifopt](https://github.com/ethz-adrl/ifopt)
Follow the installation instructions on the GitHub page, linked above.

#### [CppAD](https://github.com/coin-or/CppAD)
Download the source code and run the usual cmake commands from the repository's root directory:
```cmd
mkdir build
cmake ..
make
sudo make install
```

#### [CppADCodeGen](https://github.com/joaoleal/CppADCodeGen)
Download the source code and run the usual cmake commands from the repository's root directory.

#### [ViSP](https://github.com/lagadic/visp)
Install the prereqs and recommended 3rd partly libraries mentioned [here](https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-install-ubuntu.html), then download the source code and run the usual cmake commands from the repository's root directory. On the website, they describe creating a workspace, but you don't need to do this.

#### [pybind11](https://pybind11.readthedocs.io/en/stable/)
Install with pip, as detailed [here](https://pybind11.readthedocs.io/en/stable/installing.html). You might need to add

```cmd
$ export PATH=~/.local/bin:$PATH
```
to your $HOME/.bashrc.

#### [MuJoCo](https://mujoco.org/)
Download version 2.1.0 from [here](https://mujoco.org/download) and extract. The folder should be called mujoco210. Create a ~/.mujoco folder and move mujoco210 into ~/.mujoco. The resulting folder should be ~/.mujoco/mujoco210. Add the following to your ~/.bashrc:
```cmd
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
also, before running any MuJoCo-related code for the first time, open up a python3 terminal and import mujoco_py, so it can do some cython-related setup.

### [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse)
Install SuiteSparse using
```cmd
$ sudo apt-get install libsuitesparse-dev
```
then install scikit-sparse using
```cmd
$ pip install scikit-sparse
```
Also, make sure your scipy is up to date, so you can use the rotation functions:
```cmd
$ pip install -U scipy
```

## Software Architecture
Whenever you test code, you'll want to run a simulator node and a controller node. Currently, the two simulator nodes available are in scripts/sim_node.py and scripts/mujoco_node.py. sim_node.py simply runs forward integration using pinocchio, while mujoco_node.py uses Mujoco, and simulates contact dynamics with the client vehicle.

Note that sim_node.py doesn't publish information about the nozzle, unlike mujoco_node.py. Therefore, in conjunction with sim_node.py, you'll want to use nozzle_node.py.

There are a couple controllers in the scripts folder, and they all extend the Controller class in controller.py. The main control node is docking_node.py. control_node.py is something I was using previously for debugging things, but it's pretty out of date.

There are various yaml files in the config folder specifying parameters for the environment and controllers.
# Space Robot
Simulation for on-orbit manipulation

##  Contents
* **[Installation](#installation)**
* **[Executing](#Executing)**

## Installation

### Ubuntu enviroment
#### Dependencies
- [ROS](#ROS)
- [pinocchio](#pinocchio)
- [crocoddyl](#crocoddyl)
- [ifopt](#ifopt)
- [CppAD](#CppAD)
- [CppADCodeGen](#CppADCodeGen)
- [pybind11](#pybind11)
- [OSQP](#OSQP)
- [osqp-eigen](#osqp-eigen)
- [scikit-sparse](#scikit-sparse)

#### [ROS](https://www.ros.org/)
This code has been tested in ROS Melodic on Ubuntu 18.04, as well as ROS Noetic on Ubuntu 20.04. See the installation instructions for [Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) and [Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu).

#### [pinocchio](https://github.com/stack-of-tasks/pinocchio)
Follow the installation instructions [here](https://stack-of-tasks.github.io/pinocchio/download.html). Note, please do NOT install ros-melodic-pinocchio using apt. For whatever reason, the code doesn't work with that. Use the instructions given in the link to pinocchio's website.

#### [crocoddyl](https://github.com/loco-3d/crocoddyl)
Follow the installation instructions on the GitHub page, linked above.

#### [OSQP](https://osqp.org/)
Follow the installation instructions [here](https://osqp.org/docs/get_started/sources.html).

#### [osqp-eigen](https://github.com/robotology/osqp-eigen)
Follow the installation instructions [here](https://github.com/robotology/osqp-eigen).

#### [ifopt](https://github.com/ethz-adrl/ifopt)
Follow the installation instructions on the GitHub page, linked above.

#### [CppAD](https://github.com/coin-or/CppAD)
Download the source code and run the usual cmake commands from the repository's root directory:
```cmd
mkdir build
cmake ..
make
sudo make install
```

#### [CppADCodeGen](https://github.com/joaoleal/CppADCodeGen)
Download the source code and run the usual cmake commands from the repository's root directory.

#### [ViSP](https://github.com/lagadic/visp)
Install the prereqs and recommended 3rd partly libraries mentioned [here](https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-install-ubuntu.html), then download the source code and run the usual cmake commands from the repository's root directory. On the website, they describe creating a workspace, but you don't need to do this.

#### [pybind11](https://pybind11.readthedocs.io/en/stable/)
Install with pip, as detailed [here](https://pybind11.readthedocs.io/en/stable/installing.html). You might need to add

```cmd
$ export PATH=~/.local/bin:$PATH
```
to your $HOME/.bashrc.

#### [MuJoCo](https://mujoco.org/)
Download version 2.1.0 from [here](https://mujoco.org/download) and extract. The folder should be called mujoco210. Create a ~/.mujoco folder and move mujoco210 into ~/.mujoco. The resulting folder should be ~/.mujoco/mujoco210. Add the following to your ~/.bashrc:
```cmd
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
also, before running any MuJoCo-related code for the first time, open up a python3 terminal and import mujoco_py, so it can do some cython-related setup.

### [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse)
Install SuiteSparse using
```cmd
$ sudo apt-get install libsuitesparse-dev
```
then install scikit-sparse using
```cmd
$ pip install scikit-sparse
```
Also, make sure your scipy is up to date, so you can use the rotation functions:
```cmd
$ pip install -U scipy
```

## Software Architecture
Whenever you test code, you'll want to run a simulator node and a controller node. Currently, the two simulator nodes available are in scripts/sim_node.py and scripts/mujoco_node.py. sim_node.py simply runs forward integration using pinocchio, while mujoco_node.py uses Mujoco, and simulates contact dynamics with the client vehicle.

Note that sim_node.py doesn't publish information about the nozzle, unlike mujoco_node.py. Therefore, in conjunction with sim_node.py, you'll want to use nozzle_node.py.

There are a couple controllers in the scripts folder, and they all extend the Controller class in controller.py. The main control node is docking_node.py. control_node.py is something I was using previously for debugging things, but it's pretty out of date.

There are various yaml files in the config folder specifying parameters for the environment and controllers.

The ROS graph is shown below:

![](rosgraph.svg)

## Executing

```cmd
$ roslaunch space_robot mujoco_docking.launch
```

To execute a basin of attraction experiment,
```cmd
$ roslaunch space_robot experiment.launch
```
and set parameters in config/experiment.yaml accordingly.

To execute a moving client experiment,
```cmd
$ roslaunch space_robot moving_client_experiment.launch
```
and set parameters in config/moving_client_experiment.yaml accordingly.

To execute a moving client experiment,
```cmd
$ roslaunch space_robot moving_client_experiment.launch
```

To run the simulation with ViSP tracking, run
```cmd
$ roslaunch space_robot mujoco_docking_visp.launch
```


## Executing

```cmd
$ roslaunch space_robot mujoco_docking.launch
```

To execute a basin of attraction experiment,
```cmd
$ roslaunch space_robot experiment.launch
```
and set parameters in config/experiment.yaml accordingly.

To execute a moving client experiment,
```cmd
$ roslaunch space_robot moving_client_experiment.launch
```
and set parameters in config/moving_client_experiment.yaml accordingly.

To execute a moving client experiment,
```cmd
$ roslaunch space_robot moving_client_experiment.launch
```

To run the simulation with ViSP tracking, run
```cmd
$ roslaunch space_robot mujoco_docking_visp.launch
```
