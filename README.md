# Inverse Kinematics Experiment

This is a Python experiment for solving the inverse kinematics problem using the FABRIK algorithm with rotation. The code includes a `fabrik_solver_with_rotation()` function that takes the initial positions and orientations of the arm's joints, as well as a target position and orientation for the end effector, and returns the new positions and orientations of the arm's joints.

## Requirements

To run the code, you need to have the following packages installed:

- numpy
- matplotlib
- mpl_toolkits

You can install them using pip:

```bash
pip install -r requirements.txt
```


## Usage

To use the code, you can modify the `positions`, `orientations`, `target_position`, and `target_orientation` variables in the example usage section of the `experiment.py` file, and then run the file:

```bash
python main.py
```


This will generate two 3D plots showing the arm before and after solving the inverse kinematics problem.

Note that this is an experiment and the code is not optimized for performance or robustness.

# Co-authored by GPT-4
