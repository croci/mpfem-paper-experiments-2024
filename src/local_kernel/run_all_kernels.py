import subprocess
import datetime
import json
import io
import os
import shutil

def process_results(results):
    f = io.StringIO(results)
    times = []
    errors = []
    for line in f:
        if "Total time:" in line:
            times.append(float(line.split()[-1]))
        if "ERROR:" in line:
            errors.append(float(line.split()[-1]))

    return times,errors

temp_datestring = '-{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() )
run_dir = './temp_rundir%s/' % temp_datestring
os.makedirs(run_dir) # Throws an error if it exists
executable_filename = run_dir + 'kernel.run'

base_dir = "../ffcx/kernels/"
form_types = ["poisson", "mass"]
cell_types = ["triangle", "quadrilateral", "tetrahedron", "hexahedron"][2:]
degrees = list(range(2,11))
actcoeff_flags = ["", "_action", "_coeff", "_coeff_action"][:-2]

compiler = ["g++", "llvm", "oneapi"][0]
makefile_compiler = "" if compiler == "g++" else compiler
flush_subnormals = ["true", "false"][1]

test_info = {"compiler"         : compiler,
             "flush_subnormals" : flush_subnormals,
             "form_types"       : form_types,
             "cell_types"       : cell_types,
             "degrees"          : degrees,
             "actcoeff_flags"   : actcoeff_flags,
            }

output_dict = {"test_info" : test_info}
for form_type in form_types:
    for cell_type in cell_types:
        for actcoeff in actcoeff_flags:
            times_list = []
            error_list = []
            for degree in degrees:
                if degree > 7 and cell_type == "hexahedron":
                    continue

                kernel_name = "%s_kernel_%s_%d%s.hpp" % (form_type, cell_type, degree, actcoeff)
                kernel_file = base_dir + kernel_name

                print(kernel_name, flush=True)
                kernel = subprocess.run(["EXECUTABLE_FILENAME=%s FLUSH_SUBNORMALS=%s KERNEL_HEADER_FILE=%s make %s && %s" % (executable_filename, flush_subnormals, kernel_file, makefile_compiler, executable_filename)], shell=True, capture_output=True, check=True, text=True)
                results = kernel.stdout

                #times = [float(line.split()[-1]) for line in io.StringIO(results) if "Total time:" in line]
                times,errors = process_results(results)
                times_list.append(times)
                error_list.append(errors)
                print("Times:  ", times, flush=True)
                print("Errors: ", errors, flush=True)

            key_name = "%s_kernel_%s_%d-%d%s" % (form_type, cell_type, min(degrees), max(degrees), actcoeff)
            output_dict[key_name + "_times"] = times_list
            output_dict[key_name + "_errors"] = error_list


print("Saving to file...")
datestring = '-{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() )
with open("output_data%s.json" % datestring, 'w') as f: 
    json.dump(output_dict, f, indent=2)

print("Done.")

shutil.rmtree(run_dir)
