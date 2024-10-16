import numpy as np
from dolfinx import fem
from dolfinx.io import gmshio
from mpi4py import MPI
import basix
import io

#NOTE: tet.msh IS A HEXAHEDRAL MESH, cube.msh IS A TETRAHEDRAL MESH!!!
for which_mesh in [0,1]:
    mesh_filename = ["./tet.msh", "./cube.msh"][which_mesh]
    cell_type = ["hexahedron", "tetrahedron"][which_mesh]
    output_filename = ["hexahedron_mesh.hpp", "tetrahedron_mesh.hpp"][which_mesh]

    (mesh, cell_tags, facet_tags) = gmshio.read_from_msh(mesh_filename, gdim=3, comm=MPI.COMM_WORLD)

    el = basix.ufl.element("Lagrange", cell_type, 1, shape=(3,))
    V = fem.functionspace(mesh, el)

    ## n_cells*8*gdim tensor
    print("\nNumber of cells: %d\n" %  V.dofmap.list.shape[0])
    assert V.dofmap.list.shape[0]%16 == 0

    cell_dofs_coords = (V.tabulate_dof_coordinates()[V.dofmap.list,:]).round(decimals=15)

    if which_mesh == 1:
        cell_dofs_coords = cell_dofs_coords*5 + 1

    header = '#define N_MESH_CELLS %d\n\n' % cell_dofs_coords.shape[0]
    header += ('const std::array<double, %d> mesh_coordinates = {' % len(cell_dofs_coords.flatten()))
    footer = '};\n'

    savetxt_options = {"delimiter" : '',
                       "newline" : ",\n",
                       "comments" : "",
                       "header" : header,
                       "footer" : footer,
                       }

    np.savetxt(output_filename, cell_dofs_coords.flatten(), **savetxt_options)

    with open(output_filename, "r") as f:
        string = f.read()
        
    string = string.replace('{,\n', '{\n')
    string = string[:-7] + '};\n'

    with open(output_filename, "w") as f:
        print(string, file=f)
