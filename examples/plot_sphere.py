import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    """
    Plot sphere outputs.

    Usage:
        plot_sphere.py <files>... [--output=<dir>]

    Options:
        --output=<dir>  Output directory [default: ./frames]

    """

    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    return h5py, matplotlib, np, plt


@app.cell
def _(np):
    def build_s2_coord_vertices(phi, theta):
        phi = phi.ravel()
        phi_vert = np.concatenate([phi, [2*np.pi]])
        phi_vert -= phi_vert[1] / 2
        theta = theta.ravel()
        theta_mid = (theta[:-1] + theta[1:]) / 2
        theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
        return np.meshgrid(phi_vert, theta_vert, indexing='ij')
    return (build_s2_coord_vertices,)


@app.cell
def _(build_s2_coord_vertices, h5py, matplotlib, np, plt):
    def main(filename, start, count, output):
        """Save plot of specified tasks for given range of analysis writes."""
        # Plot settings
        task = 'vorticity'
        cmap = plt.cm.RdBu_r
        dpi = 100
        figsize = (8, 8)
        savename_func = lambda write: 'write_{:06}.png'.format(write)
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        # Plot writes
        with h5py.File(filename, mode='r') as file:
            dset = file['tasks'][task]
            phi = dset.dims[1][0][:].ravel()
            theta = dset.dims[2][0][:].ravel()
            phi_vert, theta_vert = build_s2_coord_vertices(phi, theta)
            x = np.sin(theta_vert) * np.cos(phi_vert)
            y = np.sin(theta_vert) * np.sin(phi_vert)
            z = np.cos(theta_vert)
            for index in range(start, start+count):
                data_slices = (index, slice(None), slice(None))
                data = dset[data_slices]
                clim = np.max(np.abs(data))
                norm = matplotlib.colors.Normalize(-clim, clim)
                fc = cmap(norm(data))
                #fc[:, theta.size//2, :] = [0,0,0,1]  # black equator
                if index == start:
                    surf = ax.plot_surface(x, y, z, facecolors=fc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False, zorder=5)
                    ax.set_box_aspect((1,1,1))
                    ax.set_xlim(-0.7, 0.7)
                    ax.set_ylim(-0.7, 0.7)
                    ax.set_zlim(-0.7, 0.7)
                    ax.axis('off')
                else:
                    surf.set_facecolors(fc.reshape(fc.size//4, 4))
                # Save figure
                savename = savename_func(file['scales/write_number'][index])
                savepath = output.joinpath(savename)
                fig.savefig(str(savepath), dpi=dpi)
        plt.close(fig)
    return (main,)


@app.cell
def _():
    import pathlib
    output_path = pathlib.Path('snapshots').absolute()
    # Create output directory if needed
    if not output_path.exists():
        output_path.mkdir()

    return output_path, pathlib


@app.cell
def _(pathlib):
    out_plot_path  = pathlib.Path('plots').absolute()
    # Create output directory if needed
    if not out_plot_path.exists():
        out_plot_path.mkdir()

    return (out_plot_path,)


@app.cell
def _():
    #for sp in output_path.glob('*.h5'):
    #    print(sp)
    return


@app.cell
def _(main, out_plot_path, output_path):
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    post.visit_writes(output_path.glob('*.h5'), main, output=out_plot_path)
    return


@app.cell
def _(mo, out_plot_path):
    pic_list =sorted([sp for sp in out_plot_path.glob('*.png')])
    slider = mo.ui.slider(start=0, stop=len(pic_list)-1,show_value=True, full_width= True, label="Slider", value=3)
    return pic_list, slider


@app.cell
def _(mo, pic_list, slider):
    mo.vstack([slider, mo.image(src=pic_list[slider.value], width="460px", height="460px", rounded=True)])
    return


if __name__ == "__main__":
    app.run()
