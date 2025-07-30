import marimo

__generated_with = "0.14.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Dedalus script computing the eigenmodes of waves on a clamped string.
    This script demonstrates solving a 1D eigenvalue problem and produces
    plots of the first few eigenmodes and the relative error of the eigenvalues.
    It should take just a few seconds to run (serial only).

    We use a Legendre basis to solve the EVP:
    ```
        s*u + dx(dx(u)) = 0
        u(x=0) = 0
        u(x=Lx) = 0
    ```
    where s is the eigenvalue.

    For the second derivative on a closed interval, we need two tau terms.
    Here we choose to use a first-order formulation, putting one tau term
    on an auxiliary first-order variable and another in the PDE, and lifting
    both to the first derivative basis.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import dedalus.public as d3
    import logging
    logger = logging.getLogger(__name__)

    # Parameters
    Lx = 1
    Nx = 128
    dtype = np.complex128
    return Lx, Nx, d3, dtype, np, plt


@app.cell
def _(Lx, Nx, d3, dtype):
    # Bases
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)
    xbasis = d3.Legendre(xcoord, size=Nx, bounds=(0, Lx))

    # Fields
    u = dist.Field(name='u', bases=xbasis)
    tau_1 = dist.Field(name='tau_1')
    tau_2 = dist.Field(name='tau_2')
    s = dist.Field(name='s')
    return dist, s, tau_1, tau_2, u, xbasis, xcoord


@app.cell
def _(Lx, d3, np, s, tau_1, tau_2, u, xbasis, xcoord):
    # Substitutions
    dx = lambda A: d3.Differentiate(A, xcoord)
    lift_basis = xbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    ux = dx(u) + lift(tau_1) # First-order reduction
    uxx = dx(ux) + lift(tau_2)

    # Problem
    problem = d3.EVP([u, tau_1, tau_2], eigenvalue=s, namespace=locals())
    problem.add_equation("s*u + uxx = 0")
    problem.add_equation("u(x=0) = 0")
    problem.add_equation("u(x=Lx) = 0")

    # Solve
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    evals = np.sort(solver.eigenvalues)
    n = 1 + np.arange(evals.size)
    true_evals = (n * np.pi / Lx)**2
    relative_error = np.abs(evals - true_evals) / true_evals
    return n, relative_error, solver


@app.cell
def _(n, plt, relative_error):
    # Plot
    plt.figure(figsize=(6, 4))
    plt.semilogy(n, relative_error, '.')
    plt.xlabel("eigenvalue number")
    plt.ylabel("relative eigenvalue error")
    plt.tight_layout()
    #plt.savefig("eigenvalue_error.pdf")
    #plt.savefig("eigenvalue_error.png", dpi=200)
    plt.show()
    return


@app.cell
def _(dist, np, plt, solver, u, xbasis):
    plt.figure(figsize=(6, 4))
    x = dist.local_grid(xbasis)
    for nn, idx in enumerate(np.argsort(solver.eigenvalues)[:5], start=1):
        solver.set_state(idx, solver.subsystems[0])
        ug = (u['g'] / u['g'][1]).real
        plt.plot(x, ug/np.max(np.abs(ug)), label=f"n={nn}")
    plt.xlim(0, 1)
    plt.legend(loc="lower right")
    plt.ylabel(r"mode structure")
    plt.xlabel(r"x")
    plt.tight_layout()
    #plt.savefig("eigenvectors.pdf")
    #plt.savefig("eigenvectors.png", dpi=200)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
