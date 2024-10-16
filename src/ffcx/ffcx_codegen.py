import basix.ufl
import numpy as np
import ufl
import os
import re
import shutil

import ffcx.codegeneration.jit
from ffcx.codegeneration.utils import dtype_to_c_type, dtype_to_scalar_dtype

from ufl.algorithms.compute_form_data import estimate_total_polynomial_degree,compute_form_data
from ufl.algorithms import expand_derivatives

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()

compile_args = None
dtype = np.float64

##############################################################

## bilinear forms

def poisson_form(u,v,w=None):
    if w is None:
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    else:
        a = ufl.inner(w*ufl.grad(u), ufl.grad(v)) * ufl.dx
    return a

def mass_form(u,v,w=None):
    if w is None:
        a = u*v * ufl.dx
    else:
        a = w*u*v * ufl.dx
    return a

def advection_form(u,v,w=None):
    a = u.dx(0)*v*ufl.dx
    return a

def curlcurl_form(u,v,w=None):
    a = ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx
    return a

def NS_form(u,v,w=None):
    assert w is not None
    i = 0; j = 0;
    a = u[i].dx(j)*w[j]*v[i] * ufl.dx
    #a = ufl.dot(ufl.dot(w,ufl.grad(u)), v)*ufl.dx
    return a

def test_form(u,v,w=None):
    if w is None:
        a = ufl.div(ufl.grad(u))*ufl.div(ufl.grad(v))*ufl.dx
    else:
        a = ufl.div(ufl.grad(w*u))*ufl.div(ufl.grad(v))*ufl.dx
    return a

#################################################################################

#def check_string_der(nm, dim):
#    if not any(item in nm for item in ['100', '010', '001', '10', '01']):
#        return -1
#    if dim == 3:
#        n_der = 0 if '100' in nm else (1 if '010' in nm else 2)
#    else:
#        n_der = 0 if '10' in nm else 1
#
#    return n_der

def namelist_patch(namelist, dim, eldim, domain_eldim):
    def_list = []
    for name in namelist[1:]: #skip the first one which is just "TABULATED ARRAYS:"
        nm = re.search("//.*?\[", name).group(0)[2:-1] # get actual name
        nm_eldim = int(re.search("\]\[.*?\]", name).group(0)[2:-1])

        new_name = ("FE" + nm[3:]) if nm_eldim == eldim else ("FE_DOMAIN" + nm[6:])
        def_list.append("#define %s %s\n" % (new_name, nm))

        if eldim == domain_eldim and nm_eldim == eldim: # fix for when eldim and domain_eldim are the same
            new_name = "FE_DOMAIN" + nm[6:]
            def_list.append("#define %s %s\n" % (new_name, nm))
    
    def_list.append("\n")
    return def_list

def linefilter(s, repl, filter_comments=False):
    if s[:2] == "//" and filter_comments:
        return ""

    for a,b in repl:
        s = s.replace(a,b)

    return s

def process_file(infilename, outfilename, cell_type, action, coefficient, dim, domain_nquad, domain_eldim, eldim, namelist_patch=None):
    f = open(infilename, 'r')
    outf = open(outfilename, 'w')

    repl = [("static ", ""), ("[1][1]", ""), ("[0][0]", ""), ("{{{{", "{{"), ("}}}}","}}")]

    s = ""
    while "void tabulate_tensor_integral" not in s:
        s = f.readline()

    while "{" not in s:
        s = f.readline()

    check = False
    nquad = 0
    S = []
    namelist = ["// TABULATED ARRAYS:\n"]
    while not check:
        s = f.readline()
        check = "// --------" in s or ("for" in s and s[:2] != "//")
        if not check:
            s = linefilter(s, repl, filter_comments = True)
            if nquad == 0:
                try:
                    nquad = int(re.search("\[.*?\]", s).group(0)[1:-1]) # Get NQUAD from weights vector
                    temp = re.search("_.*?\[", s).group(0)[1:-1]        # Get weird string from weights_weirdstring
                    s = s.replace("_" + temp, "_")                      # Change weights_weirdstring to weights_ 
                    repl.append(("_Q" + temp, ""))
                    repl.append(("_" + temp, "_"))
                except AttributeError: pass

            try:
                nametemp = re.search("const double .*?\[.*?\]\[.*?\]", s).group(0)[13:]
                namelist.append("//" + nametemp + "\n")
            except AttributeError: pass

        else:
            s = linefilter(s, repl)

        S.append(s)

    if domain_nquad is None:
        domain_nquad = nquad

    kernel_info_def_string = "#define CELL_TYPE_%s\n" % cell_type.upper()
    if "poisson" in outfilename: kernel_info_def_string += "#define KERNEL_TYPE_POISSON\n"
    if "mass" in outfilename: kernel_info_def_string += "#define KERNEL_TYPE_MASS\n"
    if action: kernel_info_def_string += "#define KERNEL_TYPE_ACTION\n"
    if coefficient: kernel_info_def_string += "#define KERNEL_TYPE_COEFFICIENT\n"

    sdef = "%s\n#define GDIM %d\n#define DOMAIN_NQUAD %d\n#define DOMAIN_ELDIM %d\n#define NQUAD %d\n#define ELDIM %d\n\n" % (kernel_info_def_string, dim, domain_nquad, domain_eldim, nquad, eldim)

    outf.write(sdef)

    def_list = []
    if namelist_patch is not None:
        def_list = namelist_patch(namelist, dim, eldim, domain_eldim)

    if mpiSize == 0: print(namelist, "\n", def_list, flush=True)

    if len(np.unique([int(item[4]) for item in namelist if item[4].isdigit()])) != 2:
        raise NotImplementedError("CURRENTLY ASSUMING ONLY TWO FE TYPES IN INTEGRAL (1 DOMAIN + 1 FORM). This code ignores FE# numbering in FFCx vectors!")

    S = namelist + ["\n"] + def_list + S
    for s in S[:-1]:
        outf.write(s)

    used = ""
    unused = "[[maybe_unused]]"
    s = "\n\ntemplate <typename T>\nvoid ffcx_kernel(T* restrict A, const T* restrict coordinate_dofs, %s const T* restrict w){\n" % (used if (coefficient or action) else unused)
    outf.write(s)
    outf.write(S[-1])

    repl.append(("double", "T"))

    brackets = 1
    while brackets > 0:
        s = f.readline()
        s = linefilter(s, repl)

        outf.write(s)

        if s[:2] != "//":
            brackets += s.count("{") - s.count("}")


    s = "\n\ntemplate <typename T>\nvoid ffcx_kernel_compatible(T* restrict A, const T* restrict w, %s const T* restrict c, const T* restrict coordinate_dofs, %s const int* restrict eli, %s const uint8_t* restrict qp){\n\tffcx_kernel(A, coordinate_dofs, w);\n}\n" % (unused, unused, unused)
    outf.write(s)

    f.close()
    outf.close()

def get_element_dim(el):
    try: dim = el.sub_elements[0].dim
    except IndexError: dim = el.dim

    return dim

def get_info_string(cell_type, degree, action, coefficient):

    s = ["_%s_%d" % (cell_type, degree)]

    if coefficient:
        s.append("coeff")

    if action:
        s.append("action")

    ss = "_".join(s)
    return ss

# Good ones are quads degree 3 (16 dofs per cell) and degree 7 (64 dofs per cell),
# Also hexes degree 3 (64 dofs per cell). Could do degree 1 (8 dofs) perhaps.
def generate_code(formfcn, element, action=False, coefficient=False, outfilename=None, namelist_patch=None):

    info_string = get_info_string(str(element.cell), element.degree, action, coefficient)
    outfilename = outfilename + info_string + ".hpp"

    dim = 3 if str(element.cell) in ["hexahedron", "tetrahedron"] else 2
    domain_element = basix.ufl.element("Lagrange", str(element.cell), 1, shape=(dim,))
    domain = ufl.Mesh(domain_element)
    space = ufl.FunctionSpace(domain, element)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    w = ufl.Coefficient(space) if coefficient else None

    forms = [formfcn(u,v,w)]
    if action:
        uu = ufl.Coefficient(space)
        forms[0] = ufl.action(forms[0], uu)

    compiled_forms, module, code = ffcx.codegeneration.jit.compile_forms(
        forms, options={"scalar_type": dtype}, cffi_extra_compile_args=compile_args
    )

    if outfilename is not None:
        filename = str(module).split("'")[-2].split('python')[0]
        domain_nquad = 1 if str(element.cell) in ["triangle", "tetrahedron"] else None
        domain_eldim = get_element_dim(domain_element)
        eldim = get_element_dim(element)

        shutil.copy(filename, outfilename[:-4] + "_debug" + ".hpp")
        try:
            process_file(filename, outfilename, str(element.cell), action, coefficient, dim, domain_nquad, domain_eldim, eldim, namelist_patch=namelist_patch)
        except NotImplementedError:
            #shutil.copy(filename, outfilename[:-4] + "_debug" + ".hpp")
            raise

actcoeff_flags = [(i, j) for i in range(2) for j in range(2)]
allowed = mpiSize == 1 or mpiSize == 4 or mpiSize == 8
if not allowed:
    raise ValueError("Can only run this in serial or with MPI with 4 or 8 threads.")

if mpiSize > 1:
    actcoeff_flags = [actcoeff_flags[mpiRank%4]]

cell_types = ["triangle", "quadrilateral", "tetrahedron", "hexahedron"]
degrees = np.arange(1,11, dtype=np.int64)
for cell_type in cell_types:
    for degree in degrees:
        if degree > 8 and cell_type == "hexahedron":
            continue
        for (i,j) in actcoeff_flags:
            
            if mpiSize < 8 or mpiRank//4 == 0:
                poisson_element = basix.ufl.element("Lagrange", cell_type, degree)
                generate_code(poisson_form, poisson_element, action=bool(i), coefficient=bool(j), namelist_patch=namelist_patch, outfilename='./kernels/poisson_kernel')

            if mpiSize < 8 or mpiRank//4 == 1:
                mass_element = basix.ufl.element("Lagrange", cell_type, degree)
                generate_code(mass_form, mass_element, action=bool(i), coefficient=bool(j), namelist_patch=namelist_patch, outfilename='./kernels/mass_kernel')

        if mpiRank == mpiSize-1:
            print(cell_type + "_%d: DONE!" % degree, flush=True)

        comm.barrier()

##################################################################################################

#poisson_element = basix.ufl.element("Lagrange", cell_types[3], 3)
#generate_code(poisson_form, poisson_element, action=bool(i), coefficient=bool(j), namelist_patch=namelist_patch, outfilename='poisson_kernel')

#mass_element = basix.ufl.element("Lagrange", cell_types[0], 1)
#generate_code(mass_form, mass_element, action=bool(i), coefficient=bool(j), namelist_patch=namelist_patch, outfilename='mass_kernel')

#curlcurl_element = basix.ufl.element("N1curl", "tetrahedron", 2)
#generate_code(curlcurl_form, curlcurl_element, namelist_patch=namelist_patch, outfilename='curlcurl_kernel')

#NS_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
#generate_code(NS_form, NS_element, coefficient=True, outfilename='test1')

#test_element = basix.ufl.element("Lagrange", "triangle", 4)
#generate_code(test_form, test_element, coefficient=True, outfilename='test5')
