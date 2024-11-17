
import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    if A.shape[0] != A.shape[1] or A.shape[0] != len(b):
        raise ValueError("geht nicht")

    # TODO: Perform gaussian elimination
    m = A.shape[0]
    for i in range(m):
        if use_pivoting:
            max_zeile = np.argmax(np.abs(A[i:, i])) + i
            A[[i, max_zeile]] = A[[max_zeile, i]]
            temp = b[i]
            b[i] = b[max_zeile]
            b[max_zeile] = temp
            #Zeilen tauschen
        
        if A[i][i] == 0:
                raise ValueError("noe")

        for k in range(i+1, m):
            faktor = A[k, i] / A[i, i]
            for j in range(i, m):
                A[k, j] -= faktor * A[i, j]
            b[k] -= faktor * b[i]
        #pseudocode auf wikipedia

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    if A.shape[0] != A.shape[1] or A.shape[0] != len(b):
        raise ValueError("geht nicht")
    
    rang = 0
    for i in range(len(b)):
        if np.any(A[i, :] != 0):
            rang += 1
    if rang != A.shape[0]:
        raise ValueError("noe")

    # TODO: Initialize solution vector with proper size
    x = np.zeros_like(b)

    m = A.shape[0]
    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    for i in range(m - 1, -1, -1):
        summe = 0
        for k in range(i + 1, m):
            summe += A[i, k] * x[k]
        x[i] = (b[i] - summe) / A[i, i]

    return x

def forward_subs(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = len(b)
    x = np.zeros(m)

    if L.shape[0] != L.shape[1] or L.shape[0] != len(b):
        raise ValueError("geht nicht")
    
    rang = 0
    for i in range(len(b)):
        if np.any(L[i, :] != 0):
            rang += 1
    if rang != L.shape[0]:
        raise ValueError("noe")

    for i in range(m):
        summe = 0
        if L[i, i] == 0:
            raise ValueError("noe")
        for k in range(i):
            summe += L[i, k] * x[k]
        x[i] = 1 / L[i, i] * (b[i] - summe)
    
    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L : Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape
    if n != m:
        raise ValueError("ne")
    
    """for i in range(n):
        for j in range(n):
            if M[i, j] != M[j, i]:
                raise ValueError("noe")"""
    if not np.allclose(M, M.T):
        raise ValueError("noe")


    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                summe = M[i, i] - np.sum(L[i, :]**2)
                if summe <= 0:
                    raise ValueError("nicht positiv definit")
                L[i, j] = np.sqrt(summe)
            elif i > j:
                L[i, j] = (M[i, j] - np.sum(L[i, :] * L[j, :])) * 1 / L[j, j]
            #else: 0 lassen

    transponiert = np.transpose(L)
    if not np.allclose(np.dot(L, transponiert), M):
        raise ValueError("Fehler")

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape


    if n != m: #or n != b.shape[0]
        raise ValueError("ne")
    for i in range(n):
        for j in range(n):
            if j > i and L[i, j] != 0:
                raise ValueError("noe")


    # TODO Solve the system by forward- and backsubstitution
    x = np.zeros(m)
    L_transponiert = np.transpose(L)

    z = forward_subs(L, b)
    x = back_substitution(L_transponiert, z)

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities
    
    Raised Exceptions:
    -
    Side Effects:
    -
    Forbidden:
    -
    """
    # TODO: Initialize system matrix with proper size
    L = np.zeros((n_shots * n_rays, n_grid * n_grid))

    # TODO: Initialize intensity vector
    g = np.zeros(n_shots * n_rays)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    for i in range(n_shots):
        theta = i * np.pi / n_shots
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)

        mirfaelltnichtsein = i * n_rays
        
        for j in range(n_rays):
            g[mirfaelltnichtsein + j] = intensities[j]

        for j in range(len(ray_indices)):
            L[mirfaelltnichtsein + ray_indices[j], isect_indices[j]] = lengths[j]

    # Take a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.
    #intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)


    return [L, g]

def einmal_loesen_bitte(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, z)

    return x


def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -
    Side Effects:
    -
    Forbidden:
    """
    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)

    #A = compute_cholesky(L) will nicht funktionieren

    A = np.dot(L.T, L)
    b = np.dot(L.T, g)
    B = np.linalg.cholesky(A) #cholesky Faktor

    x = einmal_loesen_bitte(B, b)

    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))

    tim = x.reshape((n_grid, n_grid))

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
