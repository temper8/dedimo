import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Dedalus script simulating the 1D Korteweg-de Vries / Burgers equation.
    This script demonstrates solving a 1D initial value problem and produces
    a space-time plot of the solution. It should take just a few seconds to
    run (serial only).

    We use a Fourier basis to solve the IVP:
    ```
        dt(u) + u*dx(u) = a*dx(dx(u)) + b*dx(dx(dx(u)))
    ```
    """
    )
    return


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run and plot kdv_burgers", kind='warn')
    run_btn
    return (run_btn,)


@app.cell
def _(mo, run_btn):
    mo.stop(not run_btn.value)
    import numpy as np
    import matplotlib.pyplot as plt
    import dedalus.public as d3
    import logging
    logger = logging.getLogger(__name__)
    return d3, logger, np, plt


@app.cell
def _(d3, np):
    # Parameters
    Lx = 10
    Nx = 1024
    a = 1e-4
    b = 2e-4
    dealias = 3/2
    stop_sim_time = 10
    timestepper = d3.SBDF2
    timestep = 2e-3
    dtype = np.float64
    return Lx, Nx, a, b, dealias, dtype, stop_sim_time, timestep, timestepper


@app.cell
def _(Lx, Nx, d3, dealias, dtype):
    # Bases
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)
    xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
    return dist, xbasis, xcoord


@app.cell
def _(dist, xbasis):
    # Fields
    u = dist.Field(name='u', bases=xbasis)
    return (u,)


@app.cell
def _(d3, xcoord):
    # Substitutions
    dx = lambda A: d3.Differentiate(A, xcoord)
    return


@app.cell
def _(d3, u):
    # Problem
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) - a*dx(dx(u)) - b*dx(dx(dx(u))) = - u*dx(u)");
    return (problem,)


@app.cell
def _(Lx, dist, np, u, xbasis):
    # Initial conditions
    x = dist.local_grid(xbasis)
    n = 20
    u['g'] = np.log(1 + np.cosh(n)**2/np.cosh(n*(x-0.2*Lx))**2) / (2*n)
    return (x,)


@app.cell
def _(problem, stop_sim_time, timestepper):
    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time
    return (solver,)


@app.cell
def _(Nx, logger, solver, timestep, u):
    # Main loop
    u_list = [u['g'].copy()]
    t_list = [solver.sim_time]
    while solver.proceed:
        solver.step(timestep)
        if solver.iteration % 1000 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
        if solver.iteration % 25 == 0:
            u_list.append(u['g'][0:Nx].copy())
            t_list.append(solver.sim_time)         
    return t_list, u_list


@app.cell
def _(Lx, a, b, np, plt, stop_sim_time, t_list, u_list, x):
    # Plot
    plt.figure(figsize=(6, 4))
    plt.pcolormesh(x.ravel(), np.array(t_list), np.array(u_list), cmap='RdBu_r', shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
    plt.xlim(0, Lx)
    plt.ylim(0, stop_sim_time)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'KdV-Burgers, (a,b)=({a},{b})')
    plt.tight_layout()
    plt.savefig('kdv_burgers.pdf')
    plt.savefig('kdv_burgers.png', dpi=200)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
