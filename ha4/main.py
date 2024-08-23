import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)
    n = x.size
    polynomial = np.poly1d(0)
    base_functions = []

    for i in range(n):
        base_polynom = np.poly1d([1])
        for j in range(n):
            if j != i:
                base_polynom = np.polymul(base_polynom, np.poly1d([1, -x[j]]) / (x[i] - x[j])) # == *=
        base_functions.append(base_polynom)

        polynomial = np.polyadd(polynomial, np.polymul(y[i], base_polynom))

    return polynomial, base_functions



def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)
    n = x.size
    spline = []
    # TODO compute piecewise interpolating cubic polynomials
    for i in range(n-1):

        A = np.array([
            [x[i]** 3, x[i]** 2, x[i], 1],
            [x[i+1]** 3, x[i+1]** 2, x[i+1], 1],
            [3 * x[i]** 2, 2 * x[i], 1, 0],
            [3 * x[i+1]** 2, 2 * x[i+1], 1, 0]
        ])
        b = np.array([y[i], y[i+1], yp[i], yp[i+1]])
        
        coeffs = np.linalg.solve(A, b)

        spline.append(np.poly1d(coeffs))

    return spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    n = x.size
    #construct linear system with natural boundary conditions
    A = np.zeros((4 * (n-1), 4 * (n-1)))
    b = np.zeros(4 * (n-1))
    spline = []

    """for i in range(n-1):
        A[4*i:4*i+4, 4*i:4*i+4] = [[x[i]**3, x[i]**2, x[i], 1],
                                    [x[i+1]**3, x[i+1]**2, x[i+1], 1],
                                    [3*x[i]**2, 2*x[i], 1, 0], 
                                    [3*x[i+1]**2, 2*x[i+1], 1, 0]]"""
    for i in range(n-1):
        A[4*i, 4*i] = x[i]** 3 #first row
        A[4*i, 4*i+1] = x[i]** 2
        A[4*i, 4*i+2] = x[i]
        A[4*i, 4*i+3] = 1

        A[4*i+1, 4*i] = x[i+1]** 3 # second
        A[4*i+1, 4*i+1] = x[i+1]** 2
        A[4*i+1, 4*i+2] = x[i+1]
        A[4*i+1, 4*i+3] = 1

        if i < n-2:
            A[4*i+2, 4*i] = 3*(x[i+1]** 2) #third
            A[4*i+2, 4*i+1] = 2 * x[i+1]
            A[4*i+2, 4*i+2] = 1
            A[4*i+2, 4*i+3] = 0
            A[4*i+2, 4*i+4] = -3 * (x[i+1]** 2)
            A[4*i+2, 4*i+5] = -2 * x[i+1]
            A[4*i+2, 4*i+6] = -1

            A[4*i+3, 4*i] = 6 * x[i+1] #fourth
            A[4*i+3, 4*i+1] = 2
            A[4*i+3, 4*i+4] = -6 * x[i+1]
            A[4*i+3, 4*i+5] = -2
            #Verbindungsstellen gleich

    for i in range(n-1):
        b[4*i] = y[i]
        b[4*i+1] = y[i+1]
        
    A[-2, :4] = [6*x[0], 2, 0, 0] #Rand
    A[-1, -4:-2] = [6*x[n-1], 2]
    b[-2:] = [0, 0]

    coeffs = np.linalg.solve(A, b)

    for i in range(n-1):
        moin = np.poly1d(coeffs[4*i:4*i+4])
        spline.append(moin)


    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    n = x.size
    # TODO: construct linear system with periodic boundary conditions
    A = np.zeros((4 * (n-1), 4 * (n-1)))
    b = np.zeros(4 * (n-1))

    for i in range(n-1):
        A[4*i, 4*i] = x[i]** 3 #first row
        A[4*i, 4*i+1] = x[i]** 2
        A[4*i, 4*i+2] = x[i]
        A[4*i, 4*i+3] = 1

        A[4*i+1, 4*i] = x[i+1]** 3 # second
        A[4*i+1, 4*i+1] = x[i+1]** 2
        A[4*i+1, 4*i+2] = x[i+1]
        A[4*i+1, 4*i+3] = 1

        if i < n-2:
            A[4*i+2, 4*i:4*i+3] = [3 * x[i+1] ** 2, 2 * x[i+1], 1]
            A[4*i+2, 4*i+4:4*i+7] = [-3 * x[i+1] ** 2, -2 * x[i+1], -1]

            A[4*i+3, 4*i:4*i+2] = [6 * x[i+1], 2]
            A[4*i+3, 4*i+4:4*i+6] = [-6 * x[i+1], -2]
            #Verbindungsstellen gleich
    for i in range(n-1):
        b[4*i] = y[i]
        b[4*i+1] = y[i+1]

    A[-2, [0, 1, 2]] = [3 * (x[0]** 2), 2 * x[0], 1]
    A[-2, [-4, -3, -2]] = [-3 * (x[n-1]** 2), -2 * x[n-1], -1]

    A[-1, [0, 1]] = [6 * x[0], 2]
    A[-1, [-4, -3]] = [-6 * x[n-1], -2]

    b[-2:] = [0, 0]

    # TODO solve linear system for the coefficients of the spline
    coeffs = np.linalg.solve(A, b)

    spline = []
    # TODO extract local interpolation coefficients from solution
    for i in range(n-1):
        moin = np.poly1d(coeffs[4*i:4*i+4])
        spline.append(moin)


    return spline


if __name__ == '__main__':

    x = np.array( [1.0, 2.0, 3.0, 4.0])
    y = np.array( [3.0, 2.0, 4.0, 1.0])

    splines = natural_cubic_interpolation( x, y)

    # # x-values to be interpolated
    # keytimes = np.linspace(0, 200, 11)
    # # y-values to be interpolated
    # keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
    #              np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5
    # keyframes.append(keyframes[0])
    # splines = []
    # for i in range(11):  # Iterate over all animated parts
    #     x = keytimes
    #     y = np.array([keyframes[k][i] for k in range(11)])
    #     spline = natural_cubic_interpolation(x, y)
    #     if len(spline) == 0:
    #         animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
    #         self.fail("Natural cubic interpolation not implemented.")
    #     splines.append(spline)

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
