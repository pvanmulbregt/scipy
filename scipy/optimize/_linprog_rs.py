#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:07:15 2018

@author: matthaberland
"""
from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.linalg import solve
from .optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from scipy.linalg import LinAlgError
from numpy.linalg.linalg import LinAlgError as LinAlgError2
from ._linprog_util import _postsolve
from .optimize import OptimizeResult

# TODO: add display?
# TODO: cythonize c_hat = c - v.dot(A); c_hat = c_hat[~bl]
# TODO: Pythranize?


def _phase_one(A, b, x0, maxiter, tol, maxupdate, mast, pivot, callback=None,
               _T_o=[]):
    """
    The purpose of phase one is to find an initial basic feasible solution
    (BFS) to the original problem.

    Generates an auxiliary problem with a trivial BFS and an objective that
    minimizes infeasibility of the original problem. Solves the auxiliary
    problem using the main simplex routine (phase two). This either yields
    a BFS to the original problem or determines that the original problem is
    infeasible. If feasible, phase one detects redundant rows in the original
    constraint matrix and removes them, then chooses additional indices as
    necessary to complete a basis/BFS for the original problem.
    """

    # x0_data = _check_x0(x0, A, b, tol)
    # if x0_data:
    #    return x0_data

    m, n = A.shape
    status = 0

    # generate auxiliary problem to get initial BFS
    A, b, c, basis, x = _generate_auxiliary_problem(A, b, x0, tol)

    # solve auxiliary problem
    phase_one_n = n
    x, basis, status, iter_k = _phase_two(c, A, x, basis, maxiter,
                                          tol, maxupdate, mast, pivot,
                                          0, callback, _T_o, phase_one_n)

    # check for infeasibility
    residual = c.dot(x)
    if status == 0 and residual > tol:
        status = 2

    # detect redundancy
    # TODO: consider removing this?
    B = A[:, basis]
    try:
        rank_revealer = solve(B, A[:, :n])
        z = _find_nonzero_rows(rank_revealer, tol)

        # eliminate redundancy
        A = A[z, :n]
        b = b[z]
    except (LinAlgError, LinAlgError2):
        status = 4

    # form solution to original problem
    x = x[:n]
    m = A.shape[0]
    basis = basis[basis < n]

    # if feasible, choose additional indices to complete basis
    if status == 0 and len(basis) < m:
        basis = _get_more_basis_columns(A, basis)

    return x, basis, A, b, residual, status, iter_k


def _check_x0(x0, A, b, tol):
    """
    Check whether x0 is an initial basic feasible solution (BFS) to the
    original problem.
    """
    if x0 is None:
        return False

    residual = np.linalg.norm(A @ x0 - b)
    if residual > tol:
        return False

    m, n = A.shape
    basis = np.where(x0)[0]
    if len(basis) > m:
        return False
    if len(basis) < m:
        basis = _get_more_basis_columns(A, basis)
    # need to choose m columns for basis
    # values of auxiliary variables might need to be nonzero to satisfy equality
    status = 0
    iter_k = 0
    return x0, basis, A, b, residual, status, iter_k

def _get_more_basis_columns(A, basis):
    """
    Called when the auxiliary problem terminates with artificial columns in
    the basis, which must be removed and replaced with non-artificial
    columns. Finds additional columns that do not make the matrix singular.
    """
    m, n = A.shape

    # options for inclusion are those that aren't already in the basis
    a = np.arange(m+n)
    bl = np.zeros(len(a), dtype=bool)
    bl[basis] = 1
    options = a[~bl]
    options = options[options < n]  # and they have to be non-artificial

    # form basis matrix
    B = np.zeros((m, m))
    B[:, 0:len(basis)] = A[:, basis]
    rank = 0  # just enter the loop

    for i in range(n):  # somewhat arbitrary, but we need another way out
        # permute the options, and take as many as needed
        new_basis = np.random.permutation(options)[:m-len(basis)]
        B[:, len(basis):] = A[:, new_basis]  # update the basis matrix
        rank = np.linalg.matrix_rank(B)      # check the rank
        if rank == m:
            break

    return np.concatenate((basis, new_basis))


def _generate_auxiliary_problem(A, b, x0, tol):
    """
    Modifies original problem to create an auxiliary problem with a trivial
    intial basic feasible solution and an objective that minimizes
    infeasibility in the original problem.

    Conceptually this is done by stacking an identity matrix on the right of
    the original constraint matrix, adding artificial variables to correspond
    with each of these new columns, and generating a cost vector that is all
    zeros except for ones corresponding with each of the new variables.

    A initial basic feasible solution is trivial: all variables are zero
    except for the artificial variables, which are set equal to the
    corresponding element of the right hand side `b`.

    Runnning the simplex method on this auxiliary problem drives all of the
    artificial variables - and thus the cost - to zero if the original problem
    is feasible. The original problem is declared infeasible otherwise.

    Much of the complexity below is to improve efficiency by using singleton
    columns in the original problem where possible and generating artificial
    variables only as necessary.
    """
    m, n = A.shape

    if x0:
        x = x0
    else:
        x = np.zeros(n)

    r = b - A@x

    A[r < 0] = -A[r < 0]  # express problem with RHS positive for trivial BFS
    b[r < 0] = -b[r < 0]  # to the auxiliary problem

    # nonzero_constraints =
    # chooses existing columns appropriate for inclusion in inital basis
    cols, rows = _select_singleton_columns(A, b)

    acols = np.arange(m-len(cols))          # indices of auxiliary columns

    # indices of corresponding rows,
    arows = np.delete(np.arange(m), rows)
    # that is, the row in each aux column with nonzero entry

    basis = np.concatenate((cols, n + acols))   # all initial basis columns
    basis_rows = np.concatenate((rows, arows))  # all intial basis rows

    # add auxiliary singleton columns
    A = np.hstack((A, np.zeros((m, m-len(cols)))))
    A[arows, n + acols] = 1

    # generate intial BFS
    x = np.zeros(m+n-len(cols))
    x[basis] = b[basis_rows]/A[basis_rows, basis]

    # generate costs to minimize infeasibility
    c = np.zeros(m+n-len(cols))
    c[n+acols] = 1

    return A, b, c, basis, x


def _select_singleton_columns(A, b):
    """
    Finds singleton columns for which the singleton entry is of the same sign
    as the right hand side; these columns are eligible for inclusion in an
    initial basis. Determines the rows in which the singleton entries are
    located. For each of these rows, returns the indices of the one singleton
    column and its corresponding row.
    """
    # find indices of all singleton columns and corresponding row indicies
    column_indices = np.nonzero(np.sum(np.abs(A) != 0, axis=0) == 1)[0]
    columns = A[:, column_indices]          # array of singleton columns
    row_indices = np.zeros(len(column_indices), dtype=int)
    nonzero_rows, nonzero_columns = np.nonzero(columns)
    row_indices[nonzero_columns] = nonzero_rows   # corresponding row indicies

    # keep only singletons with entries that have same sign as RHS
    same_sign = A[row_indices, column_indices]*b[row_indices] >= 0
    column_indices = column_indices[same_sign]
    row_indices = row_indices[same_sign]
    # this is necessary because all elements of BFS must be non-negative

    # for each row, keep only one singleton column with an entry in that row
    unique_row_indices, first_columns = np.unique(row_indices,
                                                  return_index=True)
    return column_indices[first_columns], unique_row_indices


def _find_nonzero_rows(A, tol):
    """
    Returns logical array indicating the locations of rows with at least
    one nonzero element.
    """
    return np.any(np.abs(A) > tol, axis=1)


def _select_enter_pivot(c_hat, bl, a, rule="bland", tol=1e-12):
    """
    Selects a pivot to enter the basis. Currently Bland's rule - the smallest
    index that has a negative reduced cost - is the default.
    """
    if rule.lower() == "mrc":  # index with minimum reduced cost
        return a[~bl][np.argmin(c_hat)]
    else:  # smallest index w/ negative reduced cost
        return a[~bl][c_hat < -tol][0]


def _phase_two(c, A, x, b, maxiter, tol, maxupdate, mast, pivot, iteration=0,
               callback=None, _T_o=[], phase_one_n=None):
    """
    The heart of the simplex method. Beginning with a basic feasible solution,
    moves to adjacent basic feasible solutions successively lower reduced cost.
    Terminates when there are no basic feasible solutions with lower reduced
    cost or if the problem is determined to be unbounded.

    This implementation follows the revised simplex method based on LU
    decomposition. Rather than maintaining a tableau or an inverse of the
    basis matrix, we keep a factorization of the basis matrix that allows
    efficient solution of linear systems while avoiding stability issues
    associated with inverted matrices.
    """
    m, n = A.shape
    status = 0
    a = np.arange(n)                    # indices of columns of A
    ab = np.arange(m)                   # indices of columns of B
    if maxupdate:
        # basis matrix factorization object; similar to B = A[:, b]
        B = BGLU(A, b, maxupdate, mast)
    else:
        B = LU(A, b)

    for iteration in range(iteration, iteration + maxiter):

        if callback is not None:
            if phase_one_n is not None:
                phase = 1
                x_postsolve = x[:phase_one_n]
            else:
                phase = 2
                x_postsolve = x
            x_o, fun, slack, con, _, _ = _postsolve(x_postsolve, *_T_o, tol)
            message = ""
            res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
                                  'con': con, 'nit': iteration,
                                  'phase': phase, 'complete': False,
                                  'status': 0, 'message': message,
                                  'success':False})
            callback(res)

        bl = np.zeros(len(a), dtype=bool)
        bl[b] = 1

        xb = x[b]       # basic variables
        cb = c[b]       # basic costs

        try:
            v = B.solve(cb, transposed=True)    # similar to v = solve(B.T, cb)
        except LinAlgError:
            status = 4
            break

        c_hat = c - v.dot(A)    # reduced cost
        c_hat = c_hat[~bl]
        # Above is much faster than:
        # N = A[:, ~bl]                 # slow!
        # c_hat = c[~bl] - v.T.dot(N)
        # Can we perform the mulitplication only on the nonbasic columns?

        if np.all(c_hat >= -tol):  # all reduced costs positive -> terminate
            break

        j = _select_enter_pivot(c_hat, bl, a, rule=pivot, tol=tol)
        u = B.solve(A[:, j])        # similar to u = solve(B, A[:, j])

        i = u > tol                 # if none of the u are positive, unbounded
        if not np.any(i):
            status = 3
            break

        th = xb[i]/u[i]
        l = np.argmin(th)           # implicitly selects smallest subscript
        th_star = th[l]             # step size

        x[b] = x[b] - th_star*u     # take step
        x[j] = th_star
        B.update(ab[i][l], j)       # modify basis
        b = B.b                     # similar to b[ab[i][l]] = j
    else:
        status = 1

    return x, b, status, iteration


# FIXME: is maxiter for each phase?
def _linprog_rs(c, c0, A, b, x0=None, callback=None, maxiter=1000, tol=1e-12,
                maxupdate=10, mast=False, pivot="mrc", _T_o=[],
                **unknown_options):
    """
    Solve the following linear programming problem via a two-phase
    revised simplex algorithm.::

        minimize:     c^T @ x

        subject to:  A @ x == b
                     0 <= x < oo

    Parameters
    ----------
    c : 1D array
        Coefficients of the linear objective function to be minimized.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Currently unused.)
    A : 2D array
        2D array which, when matrix-multiplied by ``x``, gives the values of
        the equality constraints at ``x``.
    b : 1D array
        1D array of values representing the RHS of each equality constraint
        (row) in ``A_eq``.
    x0 : 1D array, optional
        Starting values of the independent variables, which will be refined by
        the optimization algorithm
    callback : callable, optional (Currently unused.)

    Options
    -------
    maxiter : int
       The maximum number of iterations to perform in either phase.
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    maxupdate : int
        The maximum number of updates performed on the LU factorization.
        After this many updates is reached, the basis matrix is factorized
        from scratch.
    mast : bool
        Minimize Amortized Solve Time. If enabled, the average time to solve
        a linear system using the basis factorization is measured. Typically,
        the average solve time will decrease with each successive solve after
        initial factorization, as factorization takes much more time than the
        solve operation (and updates). Eventually, however, the updated
        factorization becomes sufficiently complex that the average solve time
        begins to increase. When this is detected, the basis is refactorized
        from scratch. Enable this option to maximize speed at the risk of
        nondeterministic behavior. Ignored if ``maxupdate`` is 0.
    pivot : "mrc" or "bland"
        Pivot rule: Minimum Reduced Cost (default) or Bland's rule. Choose
        Bland's rule if iteration limit is reached and cycling is suspected.

    Returns
    -------
    x : 1D array
        Solution vector.
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Numerical difficulties encountered
         5 : No constraints; turn presolve on

    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem.
    """

    _check_unknown_options(unknown_options)

    messages = ["Optimization terminated successfully.",
                "Iteration limit reached.",
                "The problem appears infeasible, as the phase one auxiliary "
                "problem terminated successfully with a residual of {0:.1e}, "
                "greater than the tolerance {1} required for the solution to "
                "be considered feasible. Consider increasing the tolerance to "
                "be greater than {0:.1e}. If this tolerance is unnaceptably "
                "large, the problem is likely infeasible.",
                "The problem is unbounded, as the simplex algorithm found "
                "a basic feasible solution from which there is a direction "
                "with negative reduced cost in which all decision variables "
                "increase.",
                "Numerical difficulties encountered; consider trying "
                "method='interior-point'.",
                "Problems with no constraints are trivially solved; please "
                "turn presolve on."
                ]

    # _T_o contains information for postsolve needed for callback function
    # callback function also needs `complete` argument
    # I add `complete = False` here for convenience
    _T_o = list(_T_o)
    _T_o.insert(-1, False)

    if A.size == 0:  # address test_unbounded_below_no_presolve_corrected
        return np.zeros(c.shape), 5, messages[5], 0

    x, basis, A, b, residual, status, iteration = (
        _phase_one(A, b, x0, maxiter, tol, maxupdate,
                   mast, pivot, callback, _T_o))

    if status == 0:
        x, basis, status, iteration = _phase_two(c, A, x, basis,
                                                 maxiter, tol, maxupdate,
                                                 mast, pivot, iteration,
                                                 callback, _T_o)

    return x, status, messages[status].format(residual, tol), iteration
