def linear_tnorm(y, num_points=101, plot=False):

    import numpy as np
    import matplotlib.pyplot as plt

    """
    Linear time normalization from 0 to 100%.

    Parameters:
        y (array_like): 1-D array of data points to be interpolated.
        num_points (int, optional): Number of points at the interpolation (default is 101).
        plot (bool, optional): If True, plot the original data and the interpolated data (default is False).

    Returns:
        yn (ndarray): Interpolated data points.
        tn (ndarray): New x values (from 0 to 100) for the interpolated data.
    """
    y = np.asarray(y)

    # Original time points
    t = np.linspace(0, 100, y.size)

    # New time points
    tn = np.linspace(0, 100, num_points)

    # Perform linear interpolation
    yn = np.interp(tn, t, y)

    if plot:
        plt.plot(t, y, 'o-', label='Original Data')
        plt.plot(tn, yn, '.-', label='Interpolated Data')
        plt.xlabel('[%]')
        plt.legend()
        plt.grid(True)
        plt.show()

    return yn, tn