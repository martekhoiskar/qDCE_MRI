################################################################################
######################## I M P O R T  P A C K A G E S ##########################
################################################################################

import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import matplotlib.patches as mpatches


import numpy as np
import os
from src import Modelling
from src import pharmacokinetic_modelling as pmod
import pydicom as pd
import pandas
import nibabel as nib

################################################################################
########################## P L O T T I N G   A I F  ############################
################################################################################

def IndAIF_valid_implementation(calc_dcm, patients, slices, times):
    time = np.arange(0, 60 * 3.735, 3.735)

    for patient in patients:
        plt.figure()
        AIF_slice = slices[patient]
        AIF_time = times[patient]
        #IndAIF1_path = os.path.join(calc_dcm, "PatientDatabase/{}_EMIN_{}_EMIN/AIF/IndAIF_s{}.npz".format(patient,
        #                                                                                            patient, AIF_slice))
        IndAIF1_path = os.path.join(calc_dcm, "PatientDatabase/{}_EMIN_{}_EMIN/AIF/IndAIF_t{}_.npz".format(patient,
                                                                                                          patient,
                                                                                                          AIF_time))
        IndAIF1_data = np.load(IndAIF1_path)
        IndAIF1 = IndAIF1_data["AIF"]
        popt1 = IndAIF1_data["popt"]
        std1 = IndAIF1_data["std"]
        S01 = IndAIF1_data["S0"]


        old_calc_dcm = "/Volumes/LaCie/MKH/Masteroppgave/Calculations"
        #IndAIF2_path = os.path.join(old_calc_dcm, "{}_EMIN_{}_EMIN/AIF/IndAIF_s{}.npz".format(patient, patient,
        IndAIF2_path = os.path.join(calc_dcm, "PatientDatabase/{}_EMIN_{}_EMIN/AIF/IndAIF_s{}_.npz".format(patient, patient, AIF_slice))
        IndAIF2_data = np.load(IndAIF2_path)
        IndAIF2 = IndAIF2_data["AIF"]
        popt2 = IndAIF2_data["popt"]
        std2 = IndAIF2_data["std"]
        S02 = IndAIF2_data["S0"]

        plt.plot(time, IndAIF1, linestyle="--", label="Patient: {}, new, S0: {}".format(patient, S01))
        plt.plot(time, IndAIF2, linestyle="--", label="Patient: {}, old, S0: {}".format(patient, S02))

        print("New AIF: popt = {}, std = {}".format(popt1, std1))
        print("Old AIF: popt = {}, std = {}".format(popt2, std2))

        plt.legend()
        plt.savefig("/Users/martekho/Documents/DCE-MRI/Plots/Validation/IndAIF_p_{}".format(patient))
        plt.show()


def PopAIF_valid_implementation(calc_dcm, injection_time):
    time = np.arange(0, 60 * 3.735, 3.735)
    time_rm_bl = time[injection_time:]-time[injection_time]

    popAIF1_path = os.path.join(calc_dcm, "PopAIF/popAIF_p['1001', '1002', '1005', '1007', '1008', '1011', '1012', '1016', '1019', '1020', '1022', '1023', '1026', '1031', '1032', '1038', '1039', '1041', '1042', '1044', '1045', '1048'].npz")
    popAIF2_path = os.path.join("/Volumes/LaCie/MKH/Masteroppgave/Calculations/PopAIF/popAIF_p['01', '02', '05', '07', '08', '11', '12', '16', '19', '20', '22', '23', '26', '31', '32', '38', '39', '41', '42', '44', '45', '48'].npz")

    popAIF1_data = np.load(popAIF1_path)
    popAIF2_data = np.load(popAIF2_path)
    popAIF1 = popAIF1_data["AIF"]
    popAIF2 = popAIF2_data["AIF"]
    popAIF1_rm_bl = popAIF1_data["AIF_rm_bl"]
    popAIF2_rm_bl = popAIF2_data["AIF_rm_bl"]
    popAIF1_pa = popAIF1_data["AIF_aligned_peak"]
    popAIF2_pa = popAIF2_data["AIF_aligned_peak"]
    popAIF1_pa_rm_bl = popAIF1_data["AIF_aligned_peak_rm_bl"]
    popAIF2_pa_rm_bl = popAIF2_data["AIF_aligned_peak_rm_bl"]
    popAIF1_wia = popAIF1_data["AIF_aligned_wi"]
    popAIF2_wia = popAIF2_data["AIF_aligned_wi"]
    popAIF1_wia_rm_bl = popAIF1_data["AIF_aligned_wi_rm_bl"]
    popAIF2_wia_rm_bl = popAIF2_data["AIF_aligned_wi_rm_bl"]

    popAIF1_list = [popAIF1, popAIF1_rm_bl, popAIF1_pa, popAIF1_pa_rm_bl, popAIF1_wia, popAIF1_wia_rm_bl]
    popAIF2_list = [popAIF2, popAIF2_rm_bl, popAIF2_pa, popAIF2_pa_rm_bl, popAIF2_wia, popAIF2_wia_rm_bl]

    label = ["PopAIF_bl", "PopAIF", "PopAIF_pa_bl", "PopAIF_pa", "PopAIF_wia_bl", "PopAIF_wia"]
    time_list = [time, time_rm_bl, time, time_rm_bl, time, time_rm_bl]

    for i, (popAIF1, popAIF2) in enumerate(zip(popAIF1_list, popAIF2_list)):
        plt.figure()
        plt.plot(time_list[i], popAIF1, color="tab:red", linestyle="--", label="New, {}".format(label[i]))
        plt.plot(time_list[i], popAIF2, color="tab:blue", linestyle="-.", label="Old, {}".format(label[i]))
        plt.legend()
        plt.savefig("/Users/martekho/Documents/DCE-MRI/Plots/Validation/PopAIF/{}.png".format(label[i]))
        #plt.show()

    popt1 = popAIF1_data["popt"]
    popt2 = popAIF2_data["popt"]
    popt1_rm_bl = popAIF1_data["popt_rm_bl"]
    popt2_rm_bl = popAIF2_data["popt_rm_bl"]
    popt1_pa = popAIF1_data["popt_aligned_peaks"]
    popt2_pa = popAIF2_data["popt_aligned_peaks"]
    popt1_pa_rm_bl = popAIF1_data["popt_aligned_peaks_rm_bl"]
    popt2_pa_rm_bl = popAIF2_data["popt_aligned_peaks_rm_bl"]
    popt1_wia = popAIF1_data["popt_aligned_wi"]
    popt2_wia = popAIF2_data["popt_aligned_wi"]
    popt1_wia_rm_bl = popAIF1_data["popt_aligned_wi_rm_bl"]
    popt2_wia_rm_bl = popAIF2_data["popt_aligned_wi_rm_bl"]

    popt1_list = [popt1,popt1_rm_bl,  popt1_pa, popt1_pa_rm_bl, popt1_wia, popt1_wia_rm_bl]
    popt2_list = [popt2, popt2_rm_bl, popt2_pa, popt2_pa_rm_bl, popt2_wia, popt2_wia_rm_bl]


    for i, (popAIF1, popAIF2) in enumerate(zip(popAIF1_list, popAIF2_list)):
        plt.figure()
        plt.plot(time_list[i], popAIF1, linestyle="--", label="New, {}".format(label[i]))
        plt.plot(time_list[i], Modelling.population_AIF_func(time_list[i], *popt1_list[i]), linestyle="--", label="New predicted {}".format(label[i]))
        #plt.plot(time_list[i], popAIF2, color="tab:blue", linestyle="-.", label="Old, {}".format(label[i]))
        plt.plot(time_list[i], Modelling.population_AIF_func(time_list[i], *popt2_list[i]), linestyle="--", label="Old predicted {}".format(label[i]))
        plt.legend()
        plt.savefig("/Users/martekho/Documents/DCE-MRI/Plots/Validation/PopAIF/{}.png".format(label[i]))
        # plt.show()


def IndAIFvsPopAIF(plot_dcm, calc_dcm, popAIF, AIF_slices, AIF_times, zoomed_in=True):
    popAIF_path= os.path.join(calc_dcm, "PopAIF/{}".format(popAIF))
    popAIF_data = np.load(popAIF_path)
    patients = popAIF_data['patients']
    popAIF_popt = popAIF_data['popt']
    popAIF_time = np.arange(0, 60*3.735, 1)
    indAIF_time = np.arange(0, 60*3.735, 3.735)

    colors = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:gray", "tab:purple", "tab:cyan", "tab:olive", "tab:pink", "tab:brown", "lightgreen", "magenta",
              "tab:blue", "tab:red", "tab:orange", "tab:green", "tab:gray", "tab:purple", "tab:cyan", "tab:olive", "tab:pink", "tab:brown", "lightgreen", "magenta"]
    linestyles = [":", ":", ":", ":", ":", ":", ":", ":", ":", ":", ":", ":", ":",
                  "-.", "-.", "-.", "-.", "-.", "-.", "-.", "-.", "-.", "-.", "-.", "-."]
    fig, ax1 = plt.subplots(figsize=(15, 10), layout='constrained')

    if zoomed_in == True:
        # Create a set of inset Axes: these should fill the bounding box allocated to them.
        ax2 = plt.axes([0, 0, 1, 1])
        # Manually set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax1, [0.37, 0.2, 0.58, 0.6])
        ax2.set_axes_locator(ip)
        # Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
        #mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

    # Plot the indAIF
    for i, p in enumerate(patients):
        Old_indAIF_path = os.path.join(calc_dcm, "PatientDatabase/{}_EMIN_{}_EMIN/AIF/IndAIF_s{}_.npz".format(p, p, AIF_slices[p]))
        indAIF_path = os.path.join(calc_dcm, "PatientDatabase/{}_EMIN_{}_EMIN/AIF/IndAIF_t{}_.npz".format(p, p, AIF_times[p]))
        indAIF_data = np.load(indAIF_path)
        indAIF = indAIF_data['AIF']
        ax1.plot(indAIF_time/60., indAIF*10**(3), color=colors[i], linestyle=linestyles[i], label=r"AIF$_{{ind, {}}}$".format(int(p)-1000))
        if zoomed_in == True:
            ax2.plot(indAIF_time/60., indAIF*10**(3), color=colors[i], linestyle=linestyles[i])

    #Plot the popAIF
    ax1.plot(popAIF_time/60, Modelling.population_AIF_func(popAIF_time, *popAIF_popt)*10**(3), color="black", linestyle="-", linewidth=3, label="PopAIF")
    if zoomed_in == True:
        ax2.plot(popAIF_time/60, Modelling.population_AIF_func(popAIF_time, *popAIF_popt)*10**(3), color="black", linestyle="-", linewidth=3, label="PopAIF")

    #box = ax1.get_position()
    #ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax1.set_ylabel("Concentration (mMol)", fontsize=22)
    ax1.set_xlabel("Time (min)", fontsize=20)
    lgd = ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncols=6, fontsize=20, fancybox=True, shadow=True)
    ax1.tick_params(axis="both", which="major", labelsize=18)

    pop_AIF_max = np.max(Modelling.population_AIF_func(popAIF_time, *popAIF_popt)*10**(3))
    if zoomed_in == True:
        ax2.set_ylim(0, pop_AIF_max*1.1)
        ax2.tick_params(axis="both", which="major", labelsize=16)
        #plt.tight_layout(rect=[0, 0, 1, 1.5])

    plot_dir = os.path.join(plot_dcm, "AIF")
    if os.path.exists(plot_dir) == False:
        os.makedirs(plot_dir)

    if zoomed_in == True:
        plot_path = os.path.join(plot_dir, "IndAIFvsPopAIF_{}_zoomed_.png".format(popAIF))
    else:
        plot_path = os.path.join(plot_dir, "IndAIFvsPopAIF_{}_.png".format(popAIF))
    plt.savefig(plot_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def PopAIF_comparison(plot_dcm, calc_dcm, popAIF, injection_time):
    popAIF_path = os.path.join(calc_dcm, "PopAIF/{}".format(popAIF))
    popAIF_data = np.load(popAIF_path)
    AIF_plot_dir = os.path.join(plot_dcm, "AIF")
    if os.path.exists(AIF_plot_dir) == False:
        os.makedirs(AIF_plot_dir)

    popAIF_popt_list = [popAIF_data["popt"], popAIF_data["popt_rm_bl"], popAIF_data["popt_aligned_peaks"],
                   popAIF_data["popt_aligned_peaks_rm_bl"], popAIF_data["popt_aligned_wi"],
                   popAIF_data["popt_aligned_wi_rm_bl"]]

    popAIF_std_list = [popAIF_data["std"], popAIF_data["std_rm_bl"], popAIF_data["std_aligned_peak"],
                   popAIF_data["std_aligned_peak_rm_bl"], popAIF_data["std_aligned_wi"],
                   popAIF_data["std_aligned_wi_rm_bl"]]

    colors = ["tab:blue", "tab:blue", "tab:orange", "tab:orange", "tab:olive", "tab:olive"]
    linestyles = [":", (0, (5, 10)), ":", (0, (5, 10)), ":", (0, (5, 10))]
    labels = [r"AIF$_{pop, bl}$", r"AIF$_{pop}$", r"AIF$_{pop, pa, bl}$", r"AIF$_{pop, pa}$",
              r"AIF$_{pop, wia, bl}$", r"AIF$_{pop, wia}$"]

    time = np.arange(0, 60*3.735, 1)
    popt_list = []

    plt.figure(figsize=(6, 8))
    for i, (popt, std) in enumerate(zip(popAIF_popt_list, popAIF_std_list)):
        plt.plot(time/60., Modelling.population_AIF_func(time, *popt)*10**(3), color=colors[i], linestyle=linestyles[i],
                 linewidth=1.5, label=labels[i])
        print("\nPopt for {}".format(labels[i]))
        print("A1:", popt[0] * 1000 / 60, "+/-", std[0] * 1000 / 60)
        print("A2:", popt[1] * 1000 / 60, "+/-", std[1] * 1000 / 60)
        print("Sigma1:", popt[2] / 60, "+/-", std[2] / 60)
        print("Sigma2:", popt[3] / 60, "+/-", std[3] / 60)
        print("T1:", popt[4] / 60, "+/-", std[4] / 60)
        print("T2:", popt[5] / 60, "+/-", std[5] / 60)
        print("alpha:", popt[6] * 1000, "+/-", std[6] * 1000)
        print("beta:", popt[7] * 60, "+/-", std[7] * 60)
        print("s:", popt[8] * 60, "+/-", std[8] * 60)
        print("tau:", popt[9] / 60, "+/-", std[9] / 60)

        if popt_list == []:
            popt_list.append(popt)
            popt_list = np.array(popt_list)
        else:
            popt_list = np.concatenate((popt_list, np.array([popt])), axis=0)


    plt.ylabel("Concentration (mMol)", fontsize=20)
    plt.xlabel("Time (min)", fontsize=20)
    plt.xlim(0, 2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 16})
    plt.tight_layout()
    plot_path = os.path.join(AIF_plot_dir, "popAIF_{}_comparison_small.png".format(popAIF.split()[1]))
    plt.savefig(plot_path)
    plt.show()

    # Statistics
    mean_popt = np.mean(popt_list, axis=0)
    mean_popt[0] = mean_popt[0] * 1000 / 60
    mean_popt[1] = mean_popt[1] * 1000 / 60
    mean_popt[2] = mean_popt[2] / 60
    mean_popt[3] = mean_popt[3] / 60
    mean_popt[4] = mean_popt[4] / 60
    mean_popt[5] = mean_popt[5] / 60
    mean_popt[6] = mean_popt[6] * 1000
    mean_popt[7] = mean_popt[7] * 60
    mean_popt[8] = mean_popt[8] * 60
    mean_popt[9] = mean_popt[9] / 60

    std_popt = np.std(popt_list, axis=0)
    std_popt[0] = std_popt[0] * 1000 / 60
    std_popt[1] = std_popt[1] * 1000 / 60
    std_popt[2] = std_popt[2] / 60
    std_popt[3] = std_popt[3] / 60
    std_popt[4] = std_popt[4] / 60
    std_popt[5] = std_popt[5] / 60
    std_popt[6] = std_popt[6] * 1000
    std_popt[7] = std_popt[7] * 60
    std_popt[8] = std_popt[8] * 60
    std_popt[9] = std_popt[9] / 60

    path = os.path.join(AIF_plot_dir, "popAIF_stats.txt")
    pandas.DataFrame(np.array([mean_popt, std_popt]).T).to_csv(path, header=["Mean", "\t\tStd"], sep="\t")
    print("Mean popAIF fitting parameters: \n", mean_popt)
    print("Std popAIF fitting parameters: \n", std_popt)


def CCC_estimation_and_plot(plot_dcm, parameter1, parameter2, xlabel, ylabel, title, module,
                            parameter1_name, parameter2_name, model1, model2, tumour_type,
                            patients, patient_type, **kwargs):

    CCC_module_plot_dir = os.path.join(plot_dcm, "CCCs_wo_outliers/{}/{}".format(module, tumour_type))
    if os.path.exists(CCC_module_plot_dir) == False:
        os.makedirs(CCC_module_plot_dir)

    if xlabel[-14:] == "Individual AIF":
        popAIF_type = kwargs['popAIF']
        popAIF_patients = kwargs['popAIF_patients']
        CCC_plot_path = os.path.join(CCC_module_plot_dir, "{}_{}_IndAIF_vs_{}_{}_tumours.png".format(module,
                                     parameter1_name, popAIF_type, tumour_type))

    else:
        CCC_plot_path = os.path.join(CCC_module_plot_dir, "{}_{}_{}_vs_{}_{}_{}_tumours_{}.png".format(module, model1,
                                                                            parameter1_name, model2, parameter2_name,
                                                                            tumour_type, patient_type))

    parameter1, parameter2, outliers = Modelling.remove_outliers(parameter1, parameter2)
    linreg = stats.linregress(parameter1, parameter2)
    slope = linreg.slope
    intercept = linreg.intercept
    slope_std = linreg.stderr
    intercept_std = linreg.intercept_stderr
    r_value = linreg.rvalue
    p_value = linreg.pvalue

    CCC = Modelling.concordance_correlation_coefficient(parameter1, parameter2)

    max1 = np.max(parameter1)
    max2 = np.max(parameter2)
    max = np.max(np.array([max1, max2]).flatten())
    x = np.linspace(np.min(parameter1), max, 100)

    fig, ax = plt.subplots()
    plt.scatter(parameter1, parameter2, color='gray', marker='x', s=15, label='Calculations')
    #plt.plot(x, intercept+x*slope, color='black', linestyle='-.')
    plt.plot(x, x, color='black', linestyle='-.', label='Identity line')
    #plt.plot(x, x, linestyle='-', color='black')

    props = dict(boxstyle='round', facecolor='gray', alpha=0.2)
    #plt.text(0.05, 0.95, '\n'.join(
    #    ['Slope: {:.2f}'.format(slope), 'Intercept: {:.2f}'.format(intercept), 'Pearson CC: {:.2f}'.format(r_value),
    #     'Pearson p value: {:.2f}'.format(p_value), 'CCC: {:.2f}'.format(CCC)]),
    #    bbox=props, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)

    plt.text(0.05, 0.95, '\n'.join(
        ['CCC: {:.2f}'.format(CCC)]),
             bbox=props, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)

    #plt.title(tumour_type)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(0, max1+0.1)
    plt.ylim(0, max2+0.1)
    plt.legend(loc="lower right")
    plt.savefig(CCC_plot_path)
    plt.show()

    print("{} vs {}".format(parameter1_name, parameter2_name))
    print("Pearson correlation: {} with p-value {}".format(linreg.rvalue, linreg.pvalue))
    print("Slope: {} +/- {}, Intercept: {} +/- {}".format(slope, slope_std, intercept, intercept_std))

def CCC_estimation_and_plot_for_all_popAIF(plot_dcm, IndAIF_parameter, IndAIF_rm_bl_parameter, PopAIF_parameters, chosen_popAIFs,
                                           xlabel, ylabel, title, colors, linestyles, module, parameter_name,
                                           parameter_plot_name, tumour_type, **kwargs):
    CCC_module_plot_dir = os.path.join(plot_dcm, "CCCs/{}/{}".format(module, tumour_type))
    if os.path.exists(CCC_module_plot_dir) == False:
        os.makedirs(CCC_module_plot_dir)

    CCC_plot_path = os.path.join(CCC_module_plot_dir, "{}_{}_IndAIF_vs_All_PopAIFs_{}_tumours.png".format(module,
                                                                                    parameter_name, tumour_type))

    textbox = []

    if parameter_name == 've':
        ylim = 1.0
        xlim = 1.0

    else:
        ylim = 1.2
        xlim = 1.2

    fig, ax = plt.subplots(figsize=(20, 15))
    for i, popAIF in enumerate(chosen_popAIFs):

        if popAIF[-2:] == 'Bl':
            chosen_indAIF_parameters = IndAIF_parameter
        else:
            chosen_indAIF_parameters = IndAIF_parameter
        popAIF_parameter = PopAIF_parameters[popAIF]

        if parameter_name == 've':
            chosen_indAIF_parameters, popAIF_parameter, outliers = Modelling.remove_outliers_above_one(chosen_indAIF_parameters, popAIF_parameter)

        else:
            chosen_indAIF_parameters, popAIF_parameter, outliers = Modelling.remove_outliers(chosen_indAIF_parameters, popAIF_parameter)



        linreg = stats.linregress(chosen_indAIF_parameters, popAIF_parameter)
        slope = linreg.slope
        intercept = linreg.intercept
        slope_std = linreg.stderr
        intercept_std = linreg.intercept_stderr
        r_value = linreg.rvalue
        p_value = linreg.pvalue

        CCC = Modelling.concordance_correlation_coefficient(chosen_indAIF_parameters, popAIF_parameter)
        x = np.linspace(0.01, xlim-0.01, 100)

        label = ylabel[popAIF] + " (CCC = {:.2f})".format(CCC)
        plt.scatter(chosen_indAIF_parameters, popAIF_parameter, color=colors[popAIF], marker='x', s=30)
        plt.plot(x, intercept + x * slope, color=colors[popAIF], linestyle=linestyles[popAIF], label=label, linewidth=2)
        # plt.plot(x, x, linestyle='-', color='black')

        #text = '\n'.join([r'{}'.format(ylabel[popAIF]), 'Slope: {:.2f}'.format(slope), 'Intercept: {:.2f}'.format(intercept), 'Pearson CC: {:.2f}'.format(r_value),
        #                  'Pearson p value: {:.2f}'.format(p_value), 'CCC: {:.2f}'.format(CCC)])
        #textbox.append(text)

        print("{} - IndAIF vs {}".format(parameter_name, popAIF))
        print("Pearson correlation: {} with p-value {}".format(linreg.rvalue, linreg.pvalue))
        print("Slope: {} +/- {}, Intercept: {} +/- {}".format(slope, slope_std, intercept, intercept_std))
        print("CCC: {:.2f}".format(CCC))

    props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
    #plt.text(0.05, 0.95, '\n'.join(
    #    ['Slope: {:.2f}'.format(slope), 'Intercept: {:.2f}'.format(intercept), 'Pearson CC: {:.2f}'.format(r_value),
    #     'Pearson p value: {:.2f}'.format(p_value), 'CCC: {:.2f}'.format(CCC)]),
    #         bbox=props, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    wrap_n = [15, 15, 35, 35, 45, 45]
    shift = [0.1, 0.1, 0.15, 0.15, 0.15, 0.15]
    #shift = [0.12, 0.12, 0.2, 0.2, 0.2, 0.2]
    #box = place_column_text(ax, text=textbox, xy=(0.2 , 0.2), wrap_n_list=wrap_n, bbox=False, shift=shift, ec='black', fc='w', boxstyle='square')

    if title:
        plt.title(parameter_name)

    #ax.set_ylabel(r"{} (min$^{{-1}}$) - Population AIF".format(parameter_name), fontsize=22)
    #ax.set_xlabel(r"{} (min$^{{-1}}$) - Individual AIF".format(parameter_name), fontsize=20)
    plt.ylim(0, ylim)
    plt.xlim(0, xlim)

    ax.set_ylabel(r"{} - Population AIF".format(parameter_plot_name), fontsize=22)
    ax.set_xlabel(r"{} - Individual AIF".format(parameter_plot_name), fontsize=20)
    lgd = ax.legend(loc="upper left", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=18)

    plt.tight_layout()
    #plt.savefig(CCC_plot_path, bbox_extra_artists=(box,), bbox_inches='tight')
    plt.savefig(CCC_plot_path)
    plt.show()


def place_column_text(ax, text, xy, wrap_n_list, shift, bbox=False, **kwargs):
    """ Creates a text annotation with the text in columns.
    The text columns are provided by a list of strings.
    A surrounding box can be added via bbox=True parameter.
    If so, FancyBboxPatch kwargs can be specified.

    The width of the column can be specified by wrap_n,
    the shift parameter determines how far apart the columns are.
    The axes are specified by the ax parameter.

    Requires:
    import textwrap
    import matplotlib.patches as mpatches
    """
    # place the individual text boxes, with a bbox to extract details from later
    x, y = xy
    n = 0
    text_boxes = []
    for (i, s) in zip(text, shift):
        #text = textwrap.fill(i, wrap_n)
        box = ax.text(x=x + n, y=y, s=i, va='top', ha='left',
                      bbox=dict(alpha=0, boxstyle='square,pad=0'))
        text_boxes.append(box)
        n += s

    if bbox == True:  # draw surrounding box
        # extract box data
        plt.draw()  # so we can extract real bbox data
        # first let's calulate the height of the largest bbox
        heights = []
        for box in text_boxes:
            heights.append(box.get_bbox_patch().get_extents().transformed(ax.transData.inverted()).bounds[3])
        max_height = max(heights)
        # then calculate the furthest x value of the last bbox
        end_x = text_boxes[-1].get_window_extent().transformed(ax.transData.inverted()).xmax
        # draw final
        width = end_x - x
        fancypatch_y = y - max_height
        rect = mpatches.FancyBboxPatch(xy=(x, fancypatch_y), width=width, height=max_height, **kwargs)
        ax.add_patch(rect)
    return box

def Linear_regression_and_Pearson_CC(plot_dcm, parameter1, parameter2, pearson_CC, spearman_CC,
                                     pearson_pval, spearman_pval, xlabel, ylabel, module,
                                     parameter1_name, parameter2_name, tumour_type, patients, patient_type):

    CC_module_plot_dir = os.path.join(plot_dcm, "CCs_wo_outliers/{}".format(module))
    if os.path.exists(CC_module_plot_dir) == False:
        os.makedirs(CC_module_plot_dir)

    CC_plot_path = os.path.join(CC_module_plot_dir, "{}_{}_vs_{}_{}_tumours_{}.png".format(module, parameter1_name,
                                                     parameter2_name, tumour_type, patient_type))

    # Remove nan values
    nan_indices1 = np.argwhere(np.isnan(parameter1))
    nan_indices2 = np.argwhere(np.isnan(parameter2))
    nan_indices = np.append(nan_indices1, nan_indices2)
    parameter1 = np.delete(parameter1, nan_indices)
    parameter2 = np.delete(parameter2, nan_indices)
    if len(nan_indices) != 0:
        print("There were nan values that were removed")


    linreg = stats.linregress(parameter1, parameter2)
    slope = linreg.slope
    intercept = linreg.intercept
    slope_std = linreg.stderr
    intercept_std = linreg.intercept_stderr

    x = np.linspace(np.min(parameter1), np.max(parameter1), 100)

    fig, ax = plt.subplots()
    plt.scatter(parameter1, parameter2, color='black', marker='x', s=15)
    plt.plot(x, intercept + x * slope, color='black', linestyle='-.')
    # plt.plot(x, x, linestyle='-', color='black')

    props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
    #plt.text(0.05, 0.95, '\n'.join(
    #    ['Pearson CC from linreg: {:.2f}'.format(linreg.rvalue), 'Pearson p-value: {:.2f}'.format(linreg.pvalue),
    #     'Pearson CC: {:.2f}'.format(pearson_CC), 'Pearson p-value: {:.2f}'.format(pearson_pval),
    #     'Spearman CC: {:.2f}'.format(spearman_CC), 'Spearman p-value: {:.2f}'.format(spearman_pval)]),
    #     bbox=props, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    #plt.text(0.05, 0.95, '\n'.join(
    #    ['Pearson CC: {:.2f}'.format(pearson_CC), 'Pearson p-value: {:.2f}'.format(pearson_pval),
    #     'Spearman CC: {:.2f}'.format(spearman_CC), 'Spearman p-value: {:.2f}'.format(spearman_pval)]),
    #         bbox=props, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)

    plt.text(0.05, 0.95, '\n'.join(
        ['Pearson CC: {:.2f}'.format(pearson_CC), 'Pearson p-value: {:.2f}'.format(pearson_pval)]),
         bbox=props, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(tumour_type)
    plt.savefig(CC_plot_path)
    plt.show()

    print("{} vs {}".format(parameter1_name, parameter2_name))
    print("Pearson correlation: {} with p-value {}".format(linreg.rvalue, linreg.pvalue))
    print("Pearson correlation: {} with p-value {}".format(pearson_CC, pearson_pval))
    print("Spearman correlation: {} with p-value {}".format(spearman_CC, spearman_pval))
    print("Slope: {} +/- {}, Intercept: {} +/- {}".format(slope, slope_std, intercept, intercept_std))

def heatmaps(plot_dcm, matrix, features, cmap_label, plot_label, cmap_boundaries,
             cmap, module, tumour_type, patients, color_graded, **kwargs):
    heatmap_dir = os.path.join(plot_dcm, "Heatmaps_wo_outliers/{}".format(module))
    if os.path.exists(heatmap_dir) == False:
        os.makedirs(heatmap_dir)

    heatmap_path = os.path.join(heatmap_dir, "{}_heatmap_{}_{}_tumour_{}".format(plot_label, module, tumour_type, patients))

    if color_graded:
        params_orig = {"Kep_Brix": 0, "Kel_Brix": 1, "A": 2, "Kt_TM": 3, "ve_TM": 4, "Kep_TM": 5, "Kt_ETM": 6,
                       "ve_ETM": 7, "vp_ETM": 8, "Kep_ETM": 9, "AUC60": 10, "AUC90": 11, "AUC120": 12, "TTHP": 13}
        try:
            sorted_params = kwargs['sorted_params']
        except:
            sorted_params = ["AUC60", "AUC90", "AUC120", "A", "Kt_ETM", "Kt_TM", "ve_ETM", "ve_TM", "Kep_ETM", "Kep_TM",
                             "Kep_Brix", "Kel_Brix", "vp_ETM", "TTHP"]
        sorted_params_indx = [params_orig[key] for key in sorted_params]
        rows = np.array([np.ones(14, dtype=int) * sorted_params_indx[i] for i in range(14)])
        matrix = matrix[rows, rows.T]
        features = np.array(features)[sorted_params_indx]


    fig, ax = plt.subplots(figsize=(15, 15))
    plt.imshow(matrix, cmap=cmap, vmin=cmap_boundaries[0], vmax=cmap_boundaries[1])
    ticks_list = np.arange(0, len(features), 1)
    plt.xticks(ticks_list, features, rotation='vertical')
    plt.yticks(ticks_list, features)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='both', which='minor', labelsize=28)
    cbar = plt.colorbar(label=cmap_label, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label(cmap_label, size=32, labelpad=15)

    #plt.title("{}, {}".format(module, tumour_type))
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=800)
    plt.show()



def compare_fittings(plot_loc, time, signal, model, tumour_type, **kwargs):
    #time = np.arange(0, 60*3.735, 3.735)
    start_times = [0, 4, 7]
    for start_time in start_times:
        signal_temp = signal[start_time:]
        time_temp = time[start_time:] - time[start_time]
        if model == "Brix":
            found_dicom = False
            dicom_path = kwargs['dicom']
            sek_or_min = kwargs['sek_or_min']
            injection_speed = kwargs['injection_speed'] # ml/s
            for file in os.listdir(dicom_path):
                if file[-3:] == "IMA":
                    file_path = os.path.join(dicom_path, file)
                    dicom_header = pd.dcmread(file_path)
                    bolus_volume = dicom_header[0x0018, 0x1041].value
                    bolus_volume = float(bolus_volume)  # ml
                    tau = bolus_volume / injection_speed
                    if sek_or_min == 'm' or sek_or_min == 'm' or sek_or_min == 'min' or sek_or_min == 'Min':
                        tau = tau / 60.
                    found_dicom = True
                    break

            if found_dicom == False:
                raise Exception(
                    'No bolus volume was retrieved because there was no dicom in {}'.format(dicom_path))

            params, stds, Rsq = pmod.fit_brix_model(np.array([signal_temp]), time_temp, tau) #, modified=True)
        elif model == "TM":
            Cp = kwargs['Cp']
            Cp = Cp[start_time:]
            params, stds, Rsq = pmod.fit_tofts_model(np.array([signal_temp]), Cp, time_temp, extended=False)
        else:
            Cp = kwargs['Cp']
            Cp = Cp[start_time:]
            params, stds, Rsq = pmod.fit_tofts_model(np.array([signal_temp]), Cp, time_temp, extended=True)

        plt.figure()
        plt.scatter(time_temp, signal_temp, marker="x", color="black", label="Signal")
        if model == "Brix":
            plt.plot(time_temp, pmod.Brix_model(time_temp, tau, *params), linestyle="-.", color="black",
                     label=r'Fit (K$_{{ep}}$ = {:.2f} min$^{{-1}}$, K$_{{el}}$ = {:.2f} min$^{{-1}}$, A = {:.2f})'.format(params[0][0], params[1][0], params[2][0]))
            #plt.plot(time_temp, pmod.modified_Brix_model(time_temp, *params), linestyle="-.", color="black",
            #         label=r'Fit (K$_{{ep}}$ = {:.2f} min$^{{-1}}$, K$_{{el}}$ = {:.4f} min$^{{-1}}$, A = {:.2f})'.format(
            #             params[0][0], params[1][0], params[2][0]))

        elif model == "TM":
            plt.plot(time_temp, pmod.tofts_integral(time_temp, Cp, *params), linestyle="-.", color="black",
                     label=r'Fit (K$^{{trans}}$ = {:.2f} min$^{{-1}}$, v$_{{e}}$ = {:.2f})'.format(params[0][0], params[1][0]))
        else:
            plt.plot(time_temp, pmod.ext_tofts_integral(time_temp, Cp, *params), linestyle="-.", color="black",
                     label=r'Fit (K$^{{trans}}$ = {:.2f} min$^{{-1}}$, v$_{{e}}$ = {:.2f} min$^{{-1}}$, v$_{{p}}$ = {:.2f})'.format(params[0][0], params[1][0], params[2][0]))

        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.title("{} Rsq: {}".format(model, Rsq[0]))
        plot_path = os.path.join(plot_loc, "Fitting_comparison_{}_tumour_{}_{}.png".format(model, tumour_type, start_time))
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()


def TNM_boxplot(plot_dcm, params, param_name, model, xlabel, ylabel, HPV_type, lower_lim, upper_lim):
    TNM_dir = os.path.join(plot_dcm, "TNM")
    if os.path.exists(TNM_dir) == False:
        os.makedirs(TNM_dir)

    plot_path = os.path.join(TNM_dir, "{}_{}_{}.png".format(param_name, model, HPV_type))

    fig, ax = plt.subplots(figsize=(15, 10))
    bplot = ax.boxplot(params, patch_artist=False,
                       medianprops=dict(color='olive', linewidth=4),
                       boxprops=dict(linewidth=3),
                       flierprops=dict(linewidth=3, markersize=15, markeredgewidth=3),
                       whiskerprops=dict(linewidth=3),
                       capprops=dict(linewidth=3))
    #for patch, color in zip(bplot['boxes'], colors):
    #    patch.set_facecolor(color)

    ax.set_xlabel(xlabel, fontsize=32, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=32, labelpad=20)
    ax.tick_params(axis="both", which="major", labelsize=28)
    plt.ylim(lower_lim, upper_lim)
    plt.tight_layout()
    #plt.title(HPV_type)
    plt.savefig(plot_path)
    plt.show()

def multiple_TNM_boxplot(ROI_param_list, HPV_types, parameters, xlabel, parameter_labels,
                         lower_limits, upper_limits, plot_folder):
    figure_path = os.path.join(plot_folder, "TNM/Boxplot_{}_{}.png".format(parameters, HPV_types))

    HPV_types_labels = {'T_both': 'HPV positive and negative',
                        'T_positive': 'HPV positive', 'T_negative': 'HPV negative'}

    fig, axs = plt.subplots(len(parameters), len(HPV_types), figsize=(30, 20))
    for i, parameter in enumerate(parameters):
        for j, HPV_type in enumerate(HPV_types):
            ax = axs[i, j]
            ax.boxplot(ROI_param_list[j*len(parameters)+i], patch_artist=False,
                       medianprops=dict(color='olive', linewidth=4),
                       boxprops=dict(linewidth=3),
                       flierprops=dict(linewidth=3, markersize=15, markeredgewidth=3),
                       whiskerprops=dict(linewidth=3),
                       capprops=dict(linewidth=3))

            left, width = 0.05, 0.9
            bottom, height = 0, 1
            right = left + width
            top = bottom + height

            if i == 0:
                #p = plt.Rectangle((left, bottom), width, height, fill=False)
                #p.set_transform(ax.transAxes)
                #p.set_clip_on(False)
                #ax.add_patch(p)
                ax.text(0.5 * (left + right), top + 0.05, HPV_types_labels[HPV_type], fontsize=34, weight='bold',
                        horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)

            if j == 0:
                #p = plt.Rectangle((left, bottom), width, height, fill=False)
                #p.set_transform(ax.transAxes)
                #p.set_clip_on(False)
                #ax.add_patch(p)
                ax.text(left - 0.2, 0.5 * (bottom + top), parameter_labels[parameter], fontsize=34, weight='bold',
                        rotation='vertical', horizontalalignment='right', verticalalignment='center',
                        transform=ax.transAxes)
                ax.tick_params(axis="both", which="major", labelsize=28)
            else:
                ax.tick_params(left=False, labelleft=False)

            if i == len(parameters)-1:
                ax.set_xlabel(xlabel, fontsize=32, labelpad=20)
            else:
                ax.tick_params(bottom=False, labelbottom=False)

            if i == 0 and j == 0:
                x1 = 1
                x2 = 4
                y = 3.35
                h = 0.2
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=3, color="black")
                ax.text((x1+x2)/2, y+h*0.95, "*", fontsize=40, ha="center", va="bottom", color="black")

            #if j == 0:
            #    ax.set_ylabel(parameter_labels[parameter], fontsize=32, labelpad=20)
            ax.tick_params(axis="both", which="major", labelsize=28)

            ax.set_ylim(lower_limits[parameter], upper_limits[parameter])

    plt.subplots_adjust(right=1.5, top=1.3, wspace=1, hspace=2.4)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=100)
    plt.show()

def parameter_histogram(param_map_path, tumour_segmentation, plot_dcm, patient, parameter, model, tumour, min, max):

    # Get parameter map data
    param_map = nib.load(param_map_path)
    param_map = param_map.get_fdata()

    # Get parameter data from tumour mask
    tumour_seg_flatten = tumour_segmentation.flatten()
    tumour_seg_idx = np.where(tumour_seg_flatten == True)

    param_map_flatten = param_map.flatten()
    segmented_params = param_map_flatten[tumour_seg_idx]

    # Create folder for histograms if it does not exist
    histogram_folder = os.path.join(plot_dcm, "Histogram/{}/{}".format(patient, model))
    if os.path.exists(histogram_folder) == False:
        os.makedirs(histogram_folder)

    histogram_path = os.path.join(histogram_folder, "Histogram_p{}_{}_{}_{}.png".format(patient, model, parameter, tumour))

    # Plot a histogram
    plt.figure()
    plt.hist(segmented_params, bins=50, range=(min, max))
    plt.title("{}_{}_{}".format(parameter, model, tumour))
    plt.savefig(histogram_path)
    plt.show()







