import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_arm(positions, orientations, ax):
    """Plot the arm in 3D space.

    Args:
        positions (ndarray): Array of shape (n, 3) representing the positions of the arm's joints.
        orientations (list): List of length n representing the orientations of the arm's joints as quaternions.
        ax (Axes3DSubplot): 3D axes object to plot on.
    """
    xs, ys, zs = positions.T
    ax.plot(xs, ys, zs, '-o', lw=2, markersize=8, label="Links and Joints")

    # Plot the links between joints as lines and add labels to the joints
    for i, (pos, ori) in enumerate(zip(positions, orientations)):
        end = pos + rotate_vector(np.array([0.2, 0, 0]), ori)
        ax.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]], color='r', lw=2)
        ax.text(pos[0], pos[1], pos[2], f'Joint {i}', fontsize=12)

    # Set axis labels and add legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()


def distance(a, b):
    """Calculate the Euclidean distance between two points.

    Args:
        a (ndarray): Array of shape (3,) representing a point in 3D space.
        b (ndarray): Array of shape (3,) representing another point in 3D space.

    Returns:
        float: The Euclidean distance between a and b.
    """
    return np.linalg.norm(a - b)


def normalize(v):
    """Normalize a vector.

    Args:
        v (ndarray): Array of shape (3,) representing a vector.

    Returns:
        ndarray: The normalized vector.
    """
    return v / np.linalg.norm(v)


def slerp(q1, q2, t):
    """Spherical linear interpolation (SLERP) between two quaternions.

    Args:
        q1 (quaternion): The starting quaternion.
        q2 (quaternion): The ending quaternion.
        t (float): A value between 0 and 1 representing the interpolation parameter.

    Returns:
        quaternion: The interpolated quaternion.
    """
    return quaternion.slerp(q1, q2, t)


def rotate_vector(v, q):
    """Rotate a vector by a quaternion.

    Args:
        v (ndarray): Array of shape (3,) representing the vector to rotate.
        q (quaternion): The quaternion representing the rotation.

    Returns:
        ndarray: The rotated vector.
    """
    return (q * np.quaternion(0, *v) * q.inverse()).vec


def fabrik_solver_with_rotation(positions, orientations, target_position, target_orientation, tolerance=1e-5, max_iterations=100):
    """Solve the inverse kinematics problem using the FABRIK algorithm with rotation.

    Args:
        positions (ndarray): Array of shape (n, 3) representing the initial positions of the arm's joints.
        orientations (list): List of length n representing the initial orientations of the arm's joints as quaternions.
        target_position (ndarray): Array of shape (3,) representing the target position of the arm's end effector.
        target_orientation (quaternion): The target orientation of the arm's end effector as a quaternion.
        tolerance (float, optional): The tolerance for the distance between the end effector and the target, and the absolute difference in orientation between the end effector and the target. Defaults to 1e-5.
        max_iterations (int, optional): The maximum number of iterations to run. Defaults to 100.

    Returns:
        tuple: A tuple containing the new positions and orientations of the arm's joints.
    """



    num_links = len(positions) - 1
    link_lengths = [distance(positions[i], positions[i + 1]) for i in range(num_links)]

    # If the target is unreachable, move all joints towards the target with the last joint's orientation set to the target orientation
    if distance(target_position, positions[0]) > sum(link_lengths):
        for i in range(num_links):
            direction = normalize(target_position - positions[i])
            positions[i + 1] = positions[i] + direction * link_lengths[i]
            orientations[i + 1] = target_orientation
        return positions, orientations

    iterations = 0
    while iterations < max_iterations and (distance(positions[-1], target_position) > tolerance or
                                           quaternion.absolute_distance(orientations[-1], target_orientation) > tolerance):
        # Set the end effector to the target position and orientation
        positions[-1] = target_position
        orientations[-1] = target_orientation

        # Forward reaching phase
        for i in reversed(range(num_links)):
            direction = normalize(positions[i + 1] - positions[i])
            positions[i] = positions[i + 1] - direction * link_lengths[i]
            orientations[i] = slerp(orientations[i + 1], orientations[i], link_lengths[i] / distance(positions[i], positions[i + 1]))

        # Backward reaching phase
        positions[0] = np.array([0, 0, 0])
        for i in range(num_links):
            direction = normalize(positions[i + 1] - positions[i])
            positions[i + 1] = positions[i] + direction * link_lengths[i]
            orientations[i + 1] = slerp(orientations[i], orientations[i + 1], link_lengths[i] / distance(positions[i], positions[i + 1]))

        iterations += 1

    return positions, orientations


# Example usage
positions = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 0],
    [3, 0, 0]
], dtype=np.float32)

orientations = [quaternion.one] * 4

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
plot_arm(positions, orientations, ax)
plt.show()

target_position = np.array([2, 2, 2], dtype=np.float32)
target_orientation = quaternion.from_euler_angles(0, 0, np.pi/4)

new_positions, new_orientations = fabrik_solver_with_rotation(positions, orientations, target_position, target_orientation)
print("New positions:\n", new_positions)
print("New orientations:\n", new_orientations)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
plot_arm(new_positions, new_orientations, ax)
plt.show()
