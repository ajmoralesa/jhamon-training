import numpy as np

def calculate_knee_velocity(marker3, marker4, marker5, marker6, freq=100):
    """
    Calculate knee angular velocity and acceleration from marker data.

    Parameters:
        marker3 (numpy.ndarray): Array containing X, Y coordinates of marker3.
        marker4 (numpy.ndarray): Array containing X, Y coordinates of marker4.
        marker5 (numpy.ndarray): Array containing X, Y coordinates of marker5.
        marker6 (numpy.ndarray): Array containing X, Y coordinates of marker6.
        freq (int, optional): The sampling frequency of the marker data.
                              Defaults to 100.

    Returns:
        tuple: A tuple containing three arrays:
               1. Array containing the calculated knee angles in radians (knee_rad).
               2. Array containing the calculated knee angular velocity in radians per second (knee_v_rad).
               3. Array containing the calculated knee angular acceleration in radians per second squared (knee_acc).
        Each array has the same length as the input marker arrays, except for the acceleration
        array that is one element shorter due to the differentiation process.

    Notes:
        - The marker arrays should have the same length.
        - The function calculates the knee angular velocity and acceleration based on the given marker data.
        - The knee angles are returned as an array with the same length as the input marker arrays.
        - The angular velocities and accelerations are one element shorter than the input marker arrays,
          as they are obtained from the differentiation process.

    Example:
        >>> marker3 = np.array([[x1, y1], [x2, y2], ...])
        >>> marker4 = np.array([[x1, y1], [x2, y2], ...])
        >>> marker5 = np.array([[x1, y1], [x2, y2], ...])
        >>> marker6 = np.array([[x1, y1], [x2, y2], ...])
        >>> knee_rad, knee_v_rad, knee_acc = calculate_knee_velocity(marker3, marker4, marker5, marker6)
        >>> print(knee_rad)
        [0.15 -0.32 0.12 ...]
        >>> print(knee_v_rad)
        [0.25 -0.12 0.08 ...]
        >>> print(knee_acc)
        [0.03 -0.05 0.02 ...]
    """
    knee_ang = np.zeros(len(marker3))

    for ii in range(len(marker3)):
        x1, y1 = marker3[ii]
        x2, y2 = marker4[ii]
        x3, y3 = marker5[ii]
        x4, y4 = marker6[ii]

        thigh_ang = np.arctan2(x4 - x3, y4 - y3)
        shank_ang = np.arctan2(x2 - x1, y2 - y1)

        knee_ang[ii] = shank_ang - thigh_ang

    knee_rad = np.unwrap(knee_ang)

    knee_v_rad = np.diff(knee_rad) * freq * -1
    knee_acc = np.diff(knee_v_rad)

    return knee_rad, knee_v_rad, knee_acc

def calculate_hip_velocity(marker1, marker2, marker3, marker4, freq=100):
    """
    Calculate hip angular velocity and acceleration from marker data.

    Parameters:
        marker1 (numpy.ndarray): Array containing X, Y coordinates of marker1.
        marker2 (numpy.ndarray): Array containing X, Y coordinates of marker2.
        marker3 (numpy.ndarray): Array containing X, Y coordinates of marker3.
        marker4 (numpy.ndarray): Array containing X, Y coordinates of marker4.
        freq (int, optional): The sampling frequency of the marker data.
                              Defaults to 100.

    Returns:
        tuple: A tuple containing three arrays:
               1. Array containing the calculated hip angles in radians (hip_rad).
               2. Array containing the calculated hip angular velocity in radians per second (hip_v_rad).
               3. Array containing the calculated hip angular acceleration in radians per second squared (hip_acc).
        Each array has the same length as the input marker arrays, except for the acceleration
        array that is one element shorter due to the differentiation process.
    """

    hip_ang = np.zeros(len(marker1))

    for ii in np.arange(len(marker1)):
        x1, y1, x2, y2 = (marker1[ii, 0], marker1[ii, 1],
                          marker2[ii, 0], marker2[ii, 1])
        x3, y3, x4, y4 = (marker3[ii, 0], marker3[ii, 1],
                          marker4[ii, 0], marker4[ii, 1])

        upperbody_ang = np.arctan2(x2 - x1, y2 - y1)
        thigh_ang = np.arctan2(x4 - x3, y4 - y3)
        hip_ang[ii] = upperbody_ang - thigh_ang

    hip_rad = np.unwrap(hip_ang)
    hip_v_rad = (np.diff(hip_rad) * freq) * -1
    hip_acc = np.diff(hip_v_rad)

    return hip_rad, hip_v_rad, hip_acc
