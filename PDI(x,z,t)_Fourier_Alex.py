import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import os 
    #os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    import matplotlib.pyplot as plt
    import dedalus.public as d3
    import logging
    logger = logging.getLogger(__name__)
    return d3, logger, np, plt


@app.cell
def _():
    # Parameters
    time_stop = 0.1
    timestep = 0.01
    Lx, Lz = 10, 10    # Размеры области
    Nx, Nz = 512, 16  # Разрешение сетки
    v0 = 0.5
    u1, u2 = 1.0, 1.0
    D1, D2 = 1.0, 1.0
    l_d = 0.5
    d = 5
    return Lx, Lz, Nx, Nz, d, l_d, time_stop, timestep, v0


@app.cell
def _(Lx, Lz, Nx, Nz, d3, np):
    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.complex128)
    xbasis = d3.Chebyshev(coords['x'], size=Nx, bounds=(-Lx, Lx), dealias=3/2)
    zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(-Lz, Lz), dealias=3/2)
    return coords, dist, xbasis, zbasis


@app.cell
def _(dist, xbasis, zbasis):
    # Fields
    a = dist.Field(name='a', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    v1 = dist.Field(name='v1', bases=(xbasis,zbasis))
    v2 = dist.Field(name='v2', bases=(xbasis,zbasis))
    tau1 = dist.Field(name='tau1', bases=(zbasis))  # Множители для граничных условий
    tau2 = dist.Field(name='tau2', bases=(zbasis))
    return a, b, tau1, tau2, v1, v2


@app.cell
def _(
    a,
    b,
    coords,
    d,
    d3,
    dist,
    l_d,
    np,
    tau1,
    tau2,
    v0,
    v1,
    v2,
    xbasis,
    zbasis,
):
    # Substitutions
    dx = lambda A: d3.Differentiate(A, coords['x'])
    dz = lambda A: d3.Differentiate(A, coords['z'])
    x, z = dist.local_grids(xbasis, zbasis)
    lift_xbasis = xbasis.derivative_basis(1)
    liftx = lambda A: d3.Lift(A, lift_xbasis, -1)
    ax = dx(a) + liftx(tau1) # First-order reduction
    bx = dx(b) + liftx(tau2) # First-order reduction
    az = dz(a)  # First-order reduction
    bz = dz(b) 
    azz = dz(az) 
    bzz = dz(bz) 
    v1['g'] = v0*np.exp(- 1j*((x/l_d)**2)/2 - (z/d)**2/2)
    v2['g'] = v0*np.exp(  1j*((x/l_d)**2)/2 - (z/d)**2/2)
    return x, z


@app.cell
def _(a, b, d3, tau1, tau2):
    # Problem
    problem = d3.IVP([a, b, tau1, tau2], namespace=locals())
    problem.add_equation("   1j*dt(a) + 1j*u1*ax + D1*azz - v1*b = 0")
    problem.add_equation(" - 1j*dt(b) + 1j*u2*bx + D2*bzz - v2*a = 0")
    problem.add_equation("a(x = - Lx) = 1")
    problem.add_equation("b(x = Lx) = 1")
    return (problem,)


@app.cell
def _(a, b):
    a['g'] = 1#0.01*v0
    b['g'] = 1#0.01*v0
    return


@app.cell
def _(d3, problem, time_stop):
    # Solver
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = time_stop
    return (solver,)


@app.cell
def _(a, b, logger, solver, timestep):
    # Main loop
    a_list = [a['g'].copy()]
    b_list = [b['g'].copy()]
    t_list = [solver.sim_time]
    l_t = 1
    while solver.proceed:
        solver.step(timestep)
        if solver.iteration % 100 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
        if solver.iteration % 1 == 0:
            a.change_scales(1)
            b.change_scales(1)
            a_list.append(a['g'].copy())
            b_list.append(b['g'].copy())
            t_list.append(solver.sim_time)  
            l_t = l_t + 1
    return a_list, l_t, t_list


@app.cell
def _(l_t):
    print(l_t)
    #print(np.array(a_list).shape[2])
    return


@app.cell
def _(t_list):
    t_list
    return


@app.cell
def _(Nz, a_list, np, plt, t_list, x):
    # Plot
    #t,x,z
    plt.figure(figsize=(12, 4))
    cord_x = x.real.ravel()
    i_z = int(Nz/2)
    for _t, _a in zip(t_list, a_list):
        plt.plot(cord_x, np.abs(_a[:,0]), label=f"t = {_t}")
    #plt.xlim(0, Lx)
    #plt.ylim(0, time_stop)
    plt.xlabel('x')
    plt.ylabel('Amp(a)')
    #plt.legend()
    plt.show()
    return (cord_x,)


@app.cell
def _(Nx, a_list, np, plt, t_list, z):
    cord_z = z.real.ravel()
    i_x = int(Nx/2)
    plt.figure(figsize=(6, 4))
    for _t, _a in zip(t_list, a_list):
        plt.plot(cord_z, np.abs(_a[0,:]), label=f"t = {_t}")
    #plt.xlim(0, Lx)
    #plt.ylim(0, time_stop)
    plt.xlabel('z')
    plt.ylabel('a')
    #plt.legend()
    plt.show()
    return


@app.cell
def _(cord_x, l_d, np, plt, v0):
    v1_plot = v0*np.exp(1j*((cord_x/l_d)**2)/2)
    plt.plot(cord_x, v1_plot.real)
    #plt.xlim(15,20)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
