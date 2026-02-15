import pyvista as pv

# Load your new result
mesh = pv.read(r"data\output\test1.obj")
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='silver', show_edges=True)
plotter.add_text("SEM6 Review: Phone Reconstruction Result", font_size=10)
plotter.show()