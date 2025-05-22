import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_joints_positions(filename):
    """
    Plots the 3D positions from the given data.
    
    :param data: A dictionary where each value is a list containing at least three elements representing x, y, z coordinates.
    """

    with open(filename, 'r') as file:
        data = json.load(file)

    # Prepare data for plotting
    x = []
    y = []
    z = []

    for position in data.values():
        # Assuming the first three values in the position list are x, y, z coordinates
        x.append(position[0])
        y.append(position[1])
        z.append(position[2])

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the positions
    ax.scatter(x, y, z, c='r', marker='o')

    # Label the axes
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Show the plot
    plt.show()



def plot_poses_positions(filename):
    """
    Displays and plots the positions from the given JSON file.
    
    :param filename: The path to the JSON file containing position data.
    """
    with open(filename, 'r') as file:
        data = json.load(file)

    # Prepare data for plotting
    x = []
    y = []
    z = []

    for position in data.keys():
        # Assuming the keys are strings in the format "(x, y, z)"
        coords = position.strip('()').split(',')
        x.append(float(coords[0]))
        y.append(float(coords[1]))
        z.append(float(coords[2]))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the positions
    ax.scatter(x, y, z, c='b', marker='^')

    # Label the axes
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Show the plot
    plt.show()



joints_poses_map_file_path = "joints_poses_map.json"
pose_to_joints_map_file_path = "pose_to_joints_map.json"

plot_joints_positions(joints_poses_map_file_path)
plot_poses_positions(pose_to_joints_map_file_path)
