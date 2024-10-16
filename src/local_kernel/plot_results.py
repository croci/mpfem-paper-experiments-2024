from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import json
import subprocess
import os
import sys

################################################# PYPLOT SETUP ###############################################

# change the round factor if you like
r = 1

screens = [l.split()[-3:] for l in subprocess.check_output(
    ["xrandr"]).decode("utf-8").strip().splitlines() if " connected" in l]

sizes = []
for s in screens:
    w = float(s[0].replace("mm", "")); h = float(s[2].replace("mm", "")); d = ((w**2)+(h**2))**(0.5)
    sizes.append([2*round(n/25.4, r) for n in [w, h, d]])

gnfntsz = 40
fntsz = 32
ttlfntsz = 35

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{amsmath}\usepackage{bm}\usepackage{relsize}\DeclareMathOperator{\EPS}{\mathlarger{\mathlarger{\mathlarger{\varepsilon}}}}')
plt.rc('font', family='serif', size=gnfntsz)
plt.rc('xtick', labelsize=gnfntsz)     
plt.rc('ytick', labelsize=gnfntsz)

def newfig(fign,small=False):
    figsize=sizes[0][:-1]
    if small: figsize[0] /= 2
    fig = plt.figure(fign, figsize=figsize)
    fig.patch.set_facecolor("white")
    return fig

###################################### PLOT STUFF ########################################

#               LIST OF OPTIONS:
what_to_plot = [True, # TIMES
                True, # ERRORS
                True, # n_q ERRORS
                True, # GEOMETRY ERRORS
                True] # PIE CHARTS

save = True
plot = False

basedir = "./results/"
filename = "output_data_LAST_TRY_2024-08-19.json"
results_folder_name = basedir + "figures_" + filename[:-5]
with open(basedir + filename, 'r') as f: 
    data = json.load(f)

os.makedirs(results_folder_name, exist_ok=True)

form_types = data["test_info"]["form_types"]
cell_types = data["test_info"]["cell_types"]
degree_span = data["test_info"]["degrees"]
actcoeff_flags = data["test_info"]["actcoeff_flags"]

def get_actcoeff_flag(string):
    strings = ["_coeff_action", "_coeff", "_action"] # the order here is important for the for loop
    for flag in strings:
        if flag in string: return flag[1:]

    return ""

def get_info(key, value):
    if key == "test_info":
        return None, None, None, None, None
    form_type = [item for item in form_types if item in key][0]
    cell_type = [item for item in cell_types if item in key][0]
    data_type = "times" if "times" in key else "errors"
    actcoeff_flag = get_actcoeff_flag(key)
    min_degree = min(degree_span)
    degrees = np.arange(min_degree, min_degree + np.array(value).shape[0])
    return form_type,cell_type,actcoeff_flag,data_type,degrees

if what_to_plot[0]:
    # TIMES

    labels = ["fp64", "fp32", "fp16", r"AMX-bf16", r"AVX512-bf16"]
    def_options = {"linewidth" : 5, "markersize" : 40, "markeredgewidth" : 5}
    colors = ["#404756", "#0170C7", "#00D2DD", "#D9A21B", "#d9561b"]

    plt.close("all")

    count = 0
    for include in actcoeff_flags:

        if include not in ["", "_action"]:
            print("WARNING! ONLY IMPLEMENTED FOR BILINEAR FORMS AND ACTIONS, NOT IMPLEMENTED FOR COEFFICIENTS. Skipping...")
            continue

        for (key,value) in data.items():
            form_type,cell_type,actcoeff_flag,data_type,degrees = get_info(key, value)
            if form_type is None or data_type == "errors" or actcoeff_flag != include[1:]: continue
            val = np.array(value)[:,np.array([1,2,3,6,5])] # Exclude FFCX and mixed-precision AVX
            if count%2 == 0:
                fig = newfig(count)
                ax1,ax2 = fig.subplots(1,2)
                ax = ax1
            else:
                ax = ax2

            count += 1

            ax.axhline(y=2, color=colors[1], linestyle="--", linewidth=4)
            ax.axhline(y=4, color=colors[2], linestyle="--", linewidth=4)

            for i in range(1,val.shape[1]):
                ax.plot(degrees, val[:,0]/val[:,i], ".", label=labels[i], color=colors[i], **def_options)

            form_type = form_type[0].upper() + form_type[1:]
            cell_type = cell_type[:-2] + "a"
            actcoeff_string = actcoeff_flag + " " if actcoeff_flag in ["coeff_action", "coeff", "action"] else actcoeff_flag
            ax.set_title("%s %son %s" % (form_type, actcoeff_string, cell_type))
            ax.set_xlabel("polynomial degree")
            ax.set_ylabel("speedup from fp64")
            ax.set_xticks(degrees)
            max_y = max(int(np.ceil(max(val[:,0]/val[:,-2]))), 4)
            yticks = 2**np.arange(1+int(np.ceil(np.log2(max_y))))
            if form_type != "Poisson" and include != "_action":
                yticks = yticks[1:]
                yticks[0] = 1 
            ax.set_yticks(yticks)

            ax.legend()

            if count%2 == 0:
                plt.tight_layout()
                figure_filename = results_folder_name + "/" + ("figure_timings_%s %s.pdf" % (form_type, actcoeff_string[:-1])).replace(" ", "_")
                if save: plt.savefig(figure_filename, format='pdf', dpi=600, bbox_inches = "tight")
                if plot: plt.show()

if what_to_plot[1]:
    # ERRORS

    #FIXME LABELS
    labels = ["fp16", "AVX512-bf16", "AMX-bf16", "AMX-bf16 (fp16)"]
    def_options = {"linewidth" : 5, "markersize" : 40, "markeredgewidth" : 5}
    colors = ["#404756", "#0170C7", "#00D2DD", "#D9A21B", "#d9561b"]
    #[double, single, fp16, bf16] precisions are 2**np.array([53, 24, 11, 8])
    inv_prec = 2**np.array([11, 8, 8, 8])

    plt.close("all")

    count = 0
    for include in actcoeff_flags:

        if include not in ["", "_action"]:
            print("WARNING! ONLY IMPLEMENTED FOR BILINEAR FORMS AND ACTIONS, NOT IMPLEMENTED FOR COEFFICIENTS. Skipping...")
            continue

        for (key,value) in data.items():
            form_type,cell_type,actcoeff_flag,data_type,degrees = get_info(key, value)
            if form_type is None or data_type == "times" or actcoeff_flag != include[1:]: continue
            val = np.array(value)[:,np.array([3,5,6])] 
            if count%2 == 0:
                fig = newfig(count)
                ax1,ax2 = fig.subplots(1,2)
                ax = ax1
            else:
                ax = ax2

            count += 1

            ymin = np.inf
            ymax = 0
            for i in range(val.shape[1]):
                mask = (inv_prec[i]*abs(val[:,i]))<1e4
                ax.plot(degrees[mask], inv_prec[i]*val[mask,i], "+", label=labels[i], color=colors[i+1], **def_options)

                ymin = min(min(inv_prec[i]*val[mask,i]), ymin)
                ymax = max(max(inv_prec[i]*val[mask,i]), ymax)

            form_type = form_type[0].upper() + form_type[1:]
            cell_type = cell_type[:-2] + "a"
            actcoeff_string = actcoeff_flag + " " if actcoeff_flag in ["coeff_action", "coeff", "action"] else actcoeff_flag
            ax.set_title("%s %son %s" % (form_type, actcoeff_string, cell_type))
            ax.set_xlabel("polynomial degree")
            ax.set_ylabel("relative rounding error")
            ax.set_xticks(degrees)

            yticks = ax.get_yticks()
            yticks[yticks == 0] = 1
            yticks = yticks[yticks >= 0]
            ax.set_yticks(yticks)
            
            ax.legend()

            if count%2 == 0:
                plt.tight_layout()
                figure_filename = results_folder_name + "/" + ("figure_errors_%s %s.pdf" % (form_type, actcoeff_string[:-1])).replace(" ", "_")
                if save: plt.savefig(figure_filename, format='pdf', dpi=600, bbox_inches = "tight")
                if plot: plt.show()

if what_to_plot[2]:
    # n_q ERRORS
    #NOTE: This plotting script is not great, but it works.

    #labels = ["hddh", "hddd", "sdds", "sddd"]
    labels = ["theory: $O(n_q)$", "fit:", "fp16 errors"]
    def_options = {"linewidth" : 5, "markersize" : 40, "markeredgewidth" : 5}
    colors = ["#404756", "#018dfa", "#01457b"]
    #[double, single, fp16, bf16] precisions are 2**np.array([53, 24, 11, 8])
    inv_prec = 2**np.array([11])

    degree_to_nquad_dict_hex = {2:64, 3:125, 4:216, 5:343, 6:512, 7:729, 8:1000} # hexahedron
    degree_to_nquad_dict_tet = {2:14, 3:24, 4:45, 5:74, 6:122, 7:177, 8:729, 9:1000, 10:1331} # tetrahedron
    to_nquad = lambda p,ct : degree_to_nquad_dict_hex[p] if ct == 'hexahedron' else degree_to_nquad_dict_tet[p]

    plt.close("all")

    count = 0
    for include in actcoeff_flags:

        if include not in ["", "_action"]:
            print("WARNING! ONLY IMPLEMENTED FOR BILINEAR FORMS AND ACTIONS, NOT IMPLEMENTED FOR COEFFICIENTS. Skipping...")
            continue

        for (key,value) in data.items():
            form_type,cell_type,actcoeff_flag,data_type,degrees = get_info(key, value)
            if form_type is None or data_type == "times" or actcoeff_flag != include[1:]: continue
            if cell_type != "tetrahedron" or form_type != "mass": continue
            val = np.array(value)[:,np.array([4])]
            nquads = np.array([to_nquad(p, cell_type) for p in degrees])
            if count%2 == 0:
                fig = newfig(count)
                ax1,ax2 = fig.subplots(1,2)
                ax = ax1
            else:
                ax = ax2

            count += 1

            ymin = np.inf
            ymax = 0
            for i in range(val.shape[1]):
                mask = (inv_prec[i]*abs(val[:,i]))<1e4

                fitp = np.polyfit(np.log2(nquads[mask]), np.log2(inv_prec[i]*val[mask,i]), 1)
                #print(fitp)
                fit = nquads[mask]**fitp[0]*2**fitp[1]
                rate = np.round(fitp[0],decimals=1)
                extra_label = " $O(n_q^{%.1f})$" % rate if rate != 1.0 else " $O(n_q)$"

                scale = 0.5 if count%2 == 1 else 2**fitp[1]
                #ax.plot(nquads[mask],            nquads[mask]*scale, "-", label=labels[i%3],     color=colors[i%3], **def_options)
                ax.plot(nquads[mask],                     fit, ":", label=labels[(i+1)%3] + extra_label, color=colors[(i+0)%3], **def_options)
                ax.plot(nquads[mask], inv_prec[i]*val[mask,i], ".", label=labels[(i+2)%3], color=colors[(i+1)%3], **def_options)


                ymin = min(min(inv_prec[i]*val[mask,i]), ymin)
                ymax = max(max(inv_prec[i]*val[mask,i]), ymax)

            form_type = form_type[0].upper() + form_type[1:]
            cell_type = cell_type[:-2] + "a"
            actcoeff_string = actcoeff_flag + " " if actcoeff_flag in ["coeff_action", "coeff", "action"] else actcoeff_flag
            ax.set_title("%s %son %s" % (form_type, actcoeff_string, cell_type))
            ax.set_xlabel("\# quadrature nodes $n_q$")
            ax.set_ylabel("relative rounding error")

            ax.tick_params(axis='both', which='major', labelsize=45)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)

            #ax.set_xticks([0, 100, 300, 500, 700])
            #yticks = ax.get_yticks()
            #yticks = yticks[yticks >= 0]
            #ax.set_yticks(yticks)
            
            ax.legend()

            if count%2 == 0:
                plt.tight_layout()
                figure_filename = results_folder_name + "/" + "n_q_errors.pdf"
                if save: plt.savefig(figure_filename, format='pdf', dpi=600, bbox_inches = "tight")
                if plot: plt.show()


if what_to_plot[3]:
    # GEOMETRY ERRORS

    labels = ["fp32 errors", "fit:", "theory: $\kappa_{2}(J)$"]
    def_options = {"linewidth" : 5, "markersize" : 40, "markeredgewidth" : 5}
    colors = ["#404756", "#018dfa", "#01457b"]

    plt.close("all")

    compute_cond = lambda eps : np.linalg.cond(np.array([[1, 1, 0],[1, 0, 1],[eps, 1, -1]]), 2)

    eps = 4.**-np.arange(1,8)
    errs = 2**24*np.array([2.86e-7,1.87e-6,1.46e-5,4.1e-5,6.5e-5,4.8e-4]) # fp32 relative errors, fp16 overflows too quickly to show anything

    conds = np.array([compute_cond(item) for item in eps])
    fit_params = np.polyfit(np.log10(conds[1:]), np.log10(errs), 1)
    labels[1] += " $O(\kappa_2(J))$" # No need to put fit since the rate is 1 here. % np.round(fit_params[0],decimals=1)
    #print(fit_params)

    fit = conds[1:]**fit_params[0]*10**fit_params[1]

    fig = newfig(0, small=True)
    plt.loglog(conds,     conds, '-', color=colors[0], label=labels[2], **def_options)
    plt.loglog(conds[1:],   fit, ':', color=colors[2], label=labels[1], **def_options)
    plt.loglog(conds[1:],  errs, '.', color=colors[1], label=labels[0], **def_options)

    ax = plt.gca()

    xmin = int(np.floor(np.log10(min(conds))))
    xmax = int(np.ceil(np.log10(max(conds))))
    ymin = int(np.floor(np.log10(min(min(conds), min(errs)))))
    ymax = int(np.ceil(np.log10(max(max(errs), max(conds)))))
    xrange = 10**np.arange(xmin,xmax+1 + int(xmax-xmin < ymax-ymin))
    yrange = 10**np.arange(ymin,ymax+1 + int(xmax-xmin > ymax-ymin))
    ax.set_xticks(xrange)
    ax.set_yticks(yrange)

    ax.set_title("Poisson geometry")
    ax.set_xlabel(r"condition number $\kappa_2(J)$")
    ax.set_ylabel("relative rounding error")
    ax.set_aspect('equal')
    ax.legend()

    fig.tight_layout()

    figure_filename = results_folder_name + "/" + "geometry_errors.pdf"
    if save: plt.savefig(figure_filename, format='pdf', dpi=600, bbox_inches = "tight")
    if plot: plt.show()


if what_to_plot[4]:
    # PIE CHARTS

    plt.close("all")

    colors1 = ["#2cb3ff", "#00bd7d", "#fc9345", "#698294"]
    colors2 = ["#2cb3ff", "#00bd7d","#f3be45", "#698294"]

    colors1 = [ "#357bcf", "#7e9cd6", "#b2bedc", "#e2e2e2"]
    colors2 = [ "#357bcf", "#7e9cd6", "#b2bedc", "#e2e2e2"]

    degrees1 = np.array([2, 3, 5]) # Poisson_hexahedron
    degrees2 = np.array([3, 5, 7]) # Poisson_hexahedron_action

    # percentages[i][j] is the j-th percentage of degree i
    # the third percentage for no_action is the product that forms H, while
    # for action it is the FE function evaluation
    # the fourth percentage is "other"
    percentages1 = np.array([[67, 22.5, 5.5, 0],[80, 16, 0.15, 0],[90, 9.7, 0.1, 0]])
    percentages2 = np.array([[17,   41,  30, 0],[21, 62, 11.5, 0],[45,  50, 3.5, 0]])

    percentages1[:,-1] = 100 - np.sum(percentages1,axis=1)
    percentages2[:,-1] = 100 - np.sum(percentages2,axis=1)

    labels1 = ["mat-mats", "forming $H$", "geometry", "other"]
    labels2 = ["mat-mats",    "FE feval", "geometry", "other"]

    colors = np.stack([colors1, colors2])
    degrees = [degrees1, degrees2]
    percentages = np.stack([percentages1, percentages2])
    labels = [labels1, labels2]

    fig = newfig(0)
    axes = fig.subplots(2,3)
    
    for i in range(2):
        for j in range(3):
            patches, texts = axes[i][j].pie(percentages[i][j,:], startangle=90, colors=colors[i])
            axes[i][j].set_title("$p=%d$" % degrees[i][j], y = 0, pad=-15)
            if j == 2:
                bbox = (1,0.75) if i == 0 else (1,0.25)
                title = "\% Cost " + ("(bilinear form)" if i == 0 else "(actions)")
                fig.legend(patches, labels[i], loc="outside right", bbox_to_anchor=bbox, title=title, fontsize=45)

    fig.tight_layout()
    plt.subplots_adjust(top=0.93, right=0.80)
    fig.suptitle("Poisson on hexahedra, cost distribution", fontsize=50)

    figure_filename = results_folder_name + "/" + "timings_pie_charts.pdf"
    if save: plt.savefig(figure_filename, format='pdf', dpi=600, bbox_inches = "tight")
    if plot: plt.show()
