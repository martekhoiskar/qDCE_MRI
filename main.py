################################################################################
######################## I M P O R T  P A C K A G E S ##########################
################################################################################
import os

import matplotlib.pyplot as plt

from src import usr
from src import Database
from src import PlotFunctions
from src import Modelling
from src import RaWCSV
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.mpl_axes import Axes


################################################################################
########################## M A I N  F U N C T I O N ############################
################################################################################

def main():
    # Constants
    R_Gd = 3.1  # (s*mM)^-1(https://www.mr-tip.com/serv1.php?type=db1&dbs=Clariscan%26trade%3B)
    #R_Gd = 3.7 #(M*ms)^-1 https://qibawiki.rsna.org/images/1/1f/QIBA_DCE-MRI_Profile-Stage_1-Public_Comment.pdf
    TR = 3.04
    FA = 12.
    injection_time = 4
    CA = 7
    timepoints = 60
    TA = 6
    T1 = 1550
    injection_speed = 3 # ml/s
    dt = 3.735
    time = np.arange(0, timepoints * dt, dt)
    time = time/60.
    sek_or_min = 'm'

    # Get path to DATA folder
    #data_dcm = input("Provide the directory of the DATA folder (full path) \n>>> ")
    data_dcm = "/Users/martekho/Documents/DCE-MRI/Data"

    while os.path.exists(data_dcm)==False:
        print("DATA folder does not exists. Choose again.")
        data_dcm = input("Provide the directory of the DATA folder (full path) \n>>> ")

    # Get path to CALCULATIONS folder
    #calc_dcm = input("Provide the directory of the CALCULATIONS folder (full path) \n>>> ")
    calc_dcm = "/Users/martekho/Documents/DCE-MRI/Calculations"

    while os.path.exists(calc_dcm) == False:
        print("CALCULATION folder does not exists. Choose again.")
        calc_dcm = input("Provide the directory of the CALCULATION folder (full path) \n>>> ")

    # Get path to PLOTS folder
    # calc_dcm = input("Provide the directory of the PLOTS folder (full path) \n>>> ")
    plot_dcm = "/Users/martekho/Documents/DCE-MRI/Plots"

    while os.path.exists(calc_dcm) == False:
        print("PLOTS folder does not exists. Choose again.")
        calc_dcm = input("Provide the directory of the PLOTS folder (full path) \n>>> ")

    # The patients' individual AIF slice
    AIF_slices = {'1001': 9, '1002': 11, '1003': 10, '1004': 8, '1005': 9, '1006': 8, '1007': 3, '1008': 16, '1009': 13,
                  '1011': 5, '1012': 5, '1016': 9, '1019': 11,'1032': 1, '1020': 5, '1022': 8, '1023': 6, '1026': 10,
                  '1031': 19, '1038': 3, '1039': 3, '1041': 4, '1042': 8, '1044': 6, '1045': 2, '1048': 11, '1050': 2,
                  '1051': 0, '1052': 0, '1055': 0, '1057': 0 , '1058':0, '1060':0, '1061': 0, '1062': 0, '1064': 0, '1066': 0, '1068': 0,
                  '1070': 0, '1074': 0, '1075': 0, '1077': 0, '1078': 0, '1079': 0, '1080': 0, '1081': 0, '1083': 0,
                  '1084': 0, '1086': 0, '1090': 0, '1092': 0, '1093': 0, '1096': 0,  '1097': 0, '1099': 0}

    AIF_times = {'1001': 10, '1002': 8, '1003': 9, '1005': 10, '1007': 9, '1008': 10,
                  '1011': 9, '1012': 9, '1016': 10, '1019': 8, '1020': 10, '1022': 9, '1023': 11, '1026': 10, '1031': 10,
                  '1032': 10, '1038': 9, '1039': 9, '1041': 10, '1042': 9, '1044': 10, '1045': 9, '1048': 10, '1050': 9,
                 '1051': 9, '1052': 9, '1055': 10, '1057': 10, '1058': 8, '1060': 9, '1061': 11, '1062': 8, '1064': 9, '1066': 8, '1068': 9,
                 '1070': 10, '1074': 10, '1075': 9, '1077': 10, '1078': 11, '1079': 10, '1080': 9, '1081': 9, '1083': 9,
                 '1084': 10, '1086': 10, '1090': 11, '1092': 9, '1093': 8, '1096': 9, '1097': 9, '1099': 10
                 }


    # Calculating or plotting?
    calc_or_plot = input("Do you want to do calculations (c) or plotting (p)? \n>>> ")

    if calc_or_plot=="c" or calc_or_plot=="C":
        calculations(data_dcm, calc_dcm, time, sek_or_min, AIF_slices, AIF_times, T1, R_Gd, TR, FA, injection_time, CA, TA, injection_speed, dt)

    if calc_or_plot=="p" or calc_or_plot=="P":
        plotting(data_dcm, calc_dcm, plot_dcm, time, sek_or_min, AIF_slices, AIF_times, T1, R_Gd, TR, FA, injection_time, injection_speed, CA, TA, dt)

    if calc_or_plot!="p" and calc_or_plot!="P" and calc_or_plot !="c" and calc_or_plot!="C":
        print("Calculations and/or plotting was not chosen. Process is ended.")
        return


################################################################################
########################## H E L P F U N C T I O N S ###########################
################################################################################

def calculations(data_dcm, calc_dcm, time, sek_or_min, AIF_slices, AIF_times, T1, R_Gd, TR, FA, injection_time, CA, TA, injection_speed, dt):
    # Initialise patient database
    PatientDatabase = Database.PatientDatabase(data_dcm, calc_dcm, time, sek_or_min, T1, R_Gd, TR, FA, injection_time,
                                               injection_speed, TA, dt)

    # Available patients
    available_patients = PatientDatabase.available_patients

    # Choose which calculations to do
    quant_model = usr.choose_quantity(calc_dcm)

    # Choose which patients to do calculations for
    patient_numbers = usr.choose_patients(data_dcm, quant_model, available_patients, calc=True, plot=False)

    # Add the chosen patients to PatientDatabase
    PatientDatabase.add_patients(patient_numbers['Params'], AIF_slices, AIF_times)


    # Calculate tumour concentration
    if 1 in quant_model['quantity_number']:
        for Patient in PatientDatabase.chosen_patients:

            Ct_exists = Patient.does_Ct_exist()
            if Ct_exists:
                invalid_input = True
                while invalid_input:
                    #calc_new_Ca = input("There already exists a file with the tumour concentration for patient {}. "
                                  #"Do you want to calculate it again? (y or n)\n>>> ".format(Patient.patient_number))
                    calc_new_Ca = 'y'
                    if calc_new_Ca=='y' or calc_new_Ca=='Y':
                        Patient.calculate_Ct(injection_time, R_Gd, TR, FA)
                        invalid_input = False
                    elif calc_new_Ca=='n' or calc_new_Ca=='N':
                        Patient.load_Ct()
                        invalid_input = False
                    else:
                        print("You didn't choose yes or no. Choose again.")

            else:
                Patient.calculate_Ct(injection_time, R_Gd, TR, FA)

    # Calculate individual AIF
    if 2 in quant_model['quantity_number']:
        for Patient in PatientDatabase.chosen_patients:

            IndAIF_exists = Patient.does_IndAIF_exists()
            if IndAIF_exists:
                invalid_input = True
                while invalid_input:
                    #calc_new_IndAIF = input("There allready exists a file with the individual AIF for patient {}. "
                    #                "Do you want to calculate it again? (y or n)\n>>> ".format(Patient.patient_number))
                    calc_new_IndAIF = 'y'
                    if calc_new_IndAIF == 'y' or calc_new_IndAIF == 'Y':
                        #Patient.calculate_IndAIF(T1, injection_time, R_Gd, TR, FA)
                        Patient.calculate_IndAIF()
                        invalid_input = False
                    elif calc_new_IndAIF == 'n' or calc_new_IndAIF == 'N':
                        Patient.load_IndAIF()
                        invalid_input = False
                    else:
                        print("You didn't choose yes or no. Choose again.")

            else:
                Patient.calculate_IndAIF()

    # Calculate population AIFs
    if 3 in quant_model["quantity_number"]:
        PatientDatabase.add_popAIF_patients(patient_numbers["PopAIF"], AIF_slices, AIF_times)

        popAIF_exists = PatientDatabase.does_popAIF_exists()
        if popAIF_exists:
            invalid_input = True
            while invalid_input:
                calc_new_IndAIF = input("There allready exists a file with the individual AIF. Do you want to "
                                        "calculate it again? (y or n)")
                if calc_new_IndAIF == 'y' or calc_new_IndAIF == 'Y':
                    PatientDatabase.calculate_popAIF()
                    invalid_input = False
                elif calc_new_IndAIF == 'n' or calc_new_IndAIF == 'N':
                    print("No new population AIF was calculated.")
                else:
                    print("You didn't choose yes or no. Choose again.")

        else:
            PatientDatabase.calculate_popAIF()

    # Calculate parameter maps
    if 4 in quant_model['quantity_number']:
        for Patient in PatientDatabase.chosen_patients:
            for model_number in quant_model['model_number']:
                model = quant_model['models'][model_number-1]
                Patient.calculate_pharmacokinetic_parameters(model, quant_model)

    # Calculate AUC maps
    if 5 in quant_model['quantity_number']:
        for Patient in PatientDatabase.chosen_patients:
            print("Starting to calculate AUC for patient {}".format(Patient.patient_name))
            exist_AUCs = Patient.AUC_exists(quant_model['AUC_values'])
            total_existing_AUCs = sum(exist_AUCs)
            if total_existing_AUCs>0:
                for existing_AUC, AUC_type in zip(exist_AUCs, quant_model['AUC_values']):
                    if existing_AUC:
                        invalid_input = True
                        while invalid_input:
                            #new_calculations = input("There already exists AUC{}. Do you want to calculate new AUC{} "
                            #                         "(yes (y) or no (n))?\n >>> ".format(AUC_type, AUC_type))
                            new_calculations = 'y'
                            if new_calculations == "n" or new_calculations == "N":
                                quant_model['AUC_values'].remove(AUC_type)
                                invalid_input = False
                            elif new_calculations == "y" or new_calculations == "Y":
                                invalid_input = False
                            else:
                                print("You didn't choose yes or no. Choose again.")

                if quant_model['AUC_values'] != []:
                    Patient.calculate_AUC(quant_model['AUC_values'], TA, dt)

            else:
                Patient.calculate_AUC(quant_model['AUC_values'], TA, dt)
            print("Done calculating for patient {}".format(Patient.patient_name))

    # Calculate correlations
    if 6 in quant_model['quantity_number']:
        parameters = ['Kep_Brix', 'Kel_Brix', 'A_Brix', 'Kt_TM', 've_TM',
                      'Kep_TM', 'Kt_ETM', 've_ETM', 'vp_ETM', 'Kep_ETM',
                      'AUC60', 'AUC90', 'AUC120']

        # Gathering data for the correlation calculations
        PatientDatabase.calculate_correlations(parameters)

    # ICC calculations comparing population AIFs
    if 7 in quant_model['quantity_number']:
        parameters = ['Kt_TM', 've_TM', 'Kep_TM']
        ICC_dir = os.path.join(calc_dcm, "ICC")
        if os.path.exists(ICC_dir) == False:
            os.makedirs(ICC_dir)

        for popAIF_number in quant_model['PopAIF_number']:
            popAIF_file = quant_model['PopAIFs'][popAIF_number-1]
            popAIF_file_metadata = popAIF_file.split('_')
            popAIF_patients = popAIF_file_metadata[1]
            popAIF_patients = popAIF_patients[1:]
            #popAIF_dir = os.path.join(calc_dcm, "PopAIF")
            #popAIF_path = os.path.join(popAIF_dir, popAIF_file)
            #popAIF_data = np.load(popAIF_path)
            #popAIF_patients = popAIF_data['patients']

            popAIFs = ['PopAIF', 'PopAIFrmBl', 'PopAIFpa', 'PopAIFpaRmBl', 'PopAIFwia', 'PopAIFwiaRmBl']

            for module_number in quant_model['module_number']:
                module = quant_model['modules'][module_number - 1]
                for tumour_number in quant_model['tumour_number']:
                    tumour_type = quant_model['tumour'][tumour_number - 1]
                    ICC_list = np.array([])
                    for parameter in parameters:
                        ICC_model = "twoway"
                        ICC_type = "single"
                        ICC_definition = "agreement"

                        ICC = PatientDatabase.ICC_calculation(parameter, tumour_type, module, popAIFs, popAIF_patients,
                                                              ICC_model, ICC_type, ICC_definition)
                        ICC_list = np.append(ICC_list, ICC)

                    ICC_path = os.path.join(ICC_dir, "CC_wo_outliers_{}_{}_.npz".format(module, tumour_type))
                    np.savez(ICC_path, ICC=ICC_list, patients=PatientDatabase.chosen_patients, module=module, tumour=tumour_type)

    # Calculate Pearson CC without outliers
    if 8 in quant_model['quantity_number']:
        parameters = ['Kep_Brix', 'Kel_Brix', 'A_Brix', 'Kt_TM', 've_TM',
                      'Kep_TM', 'Kt_ETM', 've_ETM', 'vp_ETM', 'Kep_ETM',
                      'AUC60', 'AUC90', 'AUC120', 'TTHP']

        for module_number in quant_model['module_number']:
            module = quant_model['modules'][module_number-1]
            for tumour_number in quant_model['tumour_number']:
                tumour_type = quant_model['tumour'][tumour_number-1]
                correlation_matrix, pvalue_matrix, slope_matrix, slope_std_matrix, \
                intercept_matrix, intercept_std_matrix, removed_outliers \
                    = PatientDatabase.calculate_correlations_wo_outliers(parameters, module, tumour_type)

                patients = [Patient.patient_name[:4] for Patient in PatientDatabase.chosen_patients]
                patients_for_path = [int(Patient.patient_name[2:4]) for Patient in PatientDatabase.chosen_patients]
                CC_dir = os.path.join(calc_dcm, "Correlation_wo_outliers")
                if os.path.exists(CC_dir) == False:
                    os.makedirs(CC_dir)

                CC_path = os.path.join(CC_dir, "CC_wo_outliers_{}_{}_p{}.npz".format(module, tumour_type,
                                                                                     patients_for_path))
                np.savez(CC_path, CC=correlation_matrix, pvalue=pvalue_matrix, slope=slope_matrix,
                         intercept=intercept_matrix, slope_std=slope_std_matrix, intercept_std=intercept_std_matrix,
                         patients=patients, module=module, tumour=tumour_type)

    # Calculate time-to-half-peak (TTHP)
    if 9 in quant_model['quantity_number']:
        for Patient in PatientDatabase.chosen_patients:
            Patient.add_tumours()
            print("Starting to calculate TTHP for patient {}".format(Patient.patient_name))
            exist_TTHP = Patient.TTHP_exists()
            if exist_TTHP:
                invalid_input = True
                while invalid_input:
                    #new_calculations = input("There already exists TTHP. Do you want to calculate new TTHP "
                    #                         "(yes (y) or no (n))?\n >>> ")
                    new_calculations = 'y'
                    if new_calculations == "y" or new_calculations == "Y":
                        Patient.calculate_TTHP()
                        invalid_input = False
                        print("New calculation for patient {} is completed".format(Patient.patient_name))
                    elif new_calculations == "n" or new_calculations == "N":
                        invalid_input = False
                        print("No new calculation for patient {}".format(Patient.patient_name))
                    else:
                        print("You didn't choose yes or no. Choose again.")
            else:
                Patient.calculate_TTHP()
                print("TTHP has been calculated for patient {}".format(Patient.patient_name))




def plotting(data_dcm, calc_dcm, plot_dcm, time, sek_or_min, AIF_slices, AIF_times, T1, R_Gd, TR, FA, injection_time, injection_speed, CA, TA, dt):
    # Initialise patient database
    PatientDatabase = Database.PatientDatabase(data_dcm, calc_dcm, time, sek_or_min, T1, R_Gd, TR, FA, injection_time, injection_speed, TA, dt)

    # Available patients
    available_patients = PatientDatabase.available_patients

    # Choose which plots to do
    plot_model = usr.choose_plots(calc_dcm)

    # Choose which patients to use while plotting
    patient_numbers = usr.choose_patients(data_dcm, {}, available_patients, calc=False, plot=True)

    # Constants
    parameter_names = {'Kep_Brix': r'K$_{ep, \ Brix}$ (min$^{-1}$)', 'Kel_Brix': r'K$_{el, \ Brix}$ (min$^{-1}$)', 'A_Brix': r'A',
                       'Kt_TM': r'K$^{trans}_{TM}$ (min$^{-1}$)', 've_TM': r'v$_{e,\ TM}$',
                       'Kep_TM': r'K$_{ep, \ TM}$ (min$^{-1}$)',
                       'Kt_ETM': r'K$^{trans}_{ETM}$ (min$^{-1}$)', 've_ETM': r'v$_{e, \ ETM}$',
                       'vp_ETM': r'v$_{p, \ ETM}$', 'Kep_ETM': r'K$_{ep, \ ETM}$ (min$^{-1}$)',
                       'AUC60': 'AUC60', 'AUC90': 'AUC90', 'AUC120': 'AUC120',
                       'TTHP': 'Time-to-half-peak'}

    module_names = {'mean': 'Mean', 'median': 'Median', 'ROI': 'ROI', 'VbyV': 'VbyV'}

    if 1 in plot_model['plot_number']:
        PlotFunctions.IndAIF_valid_implementation(calc_dcm, patient_numbers['plots'], AIF_slices, AIF_times)

    if 2 in plot_model['plot_number']:
        PlotFunctions.PopAIF_valid_implementation(calc_dcm, injection_time)

    if 3 in plot_model['plot_number']:
        PlotFunctions.validate_parameters(calc_dcm, patient_numbers['plots'])

    # Plot the population AIF together with the individual AIFs
    if 4 in plot_model['plot_number']:
        for popAIF_number in plot_model['PopAIF_number']:
            popAIF = plot_model['PopAIFs'][popAIF_number-1]
            PlotFunctions.IndAIFvsPopAIF(plot_dcm, calc_dcm, popAIF, AIF_slices, AIF_times, False)

    # Plot the different modified population AIFs against each other
    if 5 in plot_model['plot_number']:
        for popAIF_number in plot_model['PopAIF_number']:
            popAIF = plot_model['PopAIFs'][popAIF_number-1]
            PlotFunctions.PopAIF_comparison(plot_dcm, calc_dcm, popAIF, injection_time)

    # Plot comparison of parameters from IndAIF and PopAIF
    if 6 in plot_model['plot_number']:
        # Dictionary linking popAIF type to label
        popAIF_label_dict = {'PopAIF': 'Population AIF with baseline and no alignment', 'PopAIFrmBl': 'Population AIF without baseline',
                             'PopAIFpa': 'Population AIF with baseline and aligned by peaks',
                             'PopAIFpaRmBl': 'Population AIF without baseline and aligned by peaks',
                             'PopAIFwia': 'Population AIF with baseline and aligned by start of wash-in',
                             'PopAIFwiaRmBl': 'Population AIF without baseline and aligned by start of wash-in'}

        parameter_names = {'Kep_Brix': r'K$_{ep, \ Brix}$ (min$^{-1}$)', 'Kel_Brix': r'K$_{el, \ Brix}$ (min$^{-1}$)',
                           'A_Brix': r'A',
                           'Kt_TM': r'K$^{trans}_{TM}$ (min$^{-1}$)', 've_TM': r'v$_{e,\ TM}$',
                           'Kep_TM': r'K$_{ep,\ TM}$ (min$^{-1}$)',
                           'Kt_ETM': r'K$^{trans}_{ETM}$ (min$^{-1}$)', 've_ETM': r'v$_{e,\ ETM}$',
                           'vp_ETM': r'v$_{p,\ ETM}$', 'Kep_ETM': r'K$_{ep, \ ETM}$ (min$^{-1}$)',
                           'AUC60': 'AUC60 (min)', 'AUC90': 'AUC90 (min)', 'AUC120': 'AUC120 (min)'}

        # Add the chosen patients to PatientDatabase
        PatientDatabase.add_patients(patient_numbers['plots'], AIF_slices, AIF_times)

        # Choose which parameters to compare and do CCC plots for
        chosen_parameter_numbers = plot_model['param_number']
        chosen_parameters = [plot_model['parameters'][n - 1] for n in chosen_parameter_numbers]

        # Choose which population AIF to plot for
        chosen_popAIF_type_numbers = plot_model['PopAIF_type_number']
        popAIFs = [plot_model['PopAIF_type'][n-1] for n in chosen_popAIF_type_numbers]

        # Choose whether you want to plot just nodal, primary or both tumour types
        chosen_tumour_type_number = plot_model['tumour_number']
        tumour_types = [plot_model['tumour type'][n-1] for n in chosen_tumour_type_number]

        # Choose module (VbyV, ROI etc)
        chosen_module_number = plot_model['module_number']
        modules = [plot_model['modules'][n-1] for n in chosen_module_number]

        # Chosen population AIFs
        chosen_popAIF_number = plot_model['PopAIF_number']
        popAIF_files = [plot_model['PopAIFs'][n-1] for n in chosen_popAIF_number]

        for popAIF_file in popAIF_files:
            # The patients the population AIF is based on
            popAIF_metadata = popAIF_file.split('_')
            popAIF_patients = popAIF_metadata[1]

            for parameter_metadata in chosen_parameters:
                parameter_metadata_list = parameter_metadata.split('_')
                parameter = parameter_metadata_list[0]
                model = parameter_metadata_list[1]
                for module in modules:
                    for tumour_type in tumour_types:
                        IndAIF_parameter_list = PatientDatabase.retrieve_IndAIF_and_PopAIF_parameters_from_patients(parameter,
                                                            model, "IndAIF", popAIF_patients, tumour_type, module)

                        for popAIF_type in popAIFs:
                            # Gathering data for the correlation calculations
                            PopAIF_parameter_list = PatientDatabase.retrieve_IndAIF_and_PopAIF_parameters_from_patients(parameter,
                                                            model, popAIF_type, popAIF_patients, tumour_type, module)

                            PlotFunctions.CCC_estimation_and_plot(plot_dcm, IndAIF_parameter_list,
                                          PopAIF_parameter_list,
                                          "{} - Individual AIF".format(parameter_names[parameter_metadata]),
                                          "{} - Population AIF".format(parameter_names[parameter_metadata]), False,
                                          module, parameter, parameter, model, model, tumour_type, patient_numbers['plots'],
                                          "IndAIFandPopAIF", popAIF_patients=popAIF_patients, popAIF=popAIF_type,
                                          paramter_name = parameter, Pearson=True)


    if 7 in plot_model['plot_number']:
        # Add the chosen patients to PatientDatabase
        PatientDatabase.add_patients(patient_numbers['plots'], AIF_slices, AIF_times)

        # Choose which parameters to compare and do CCC plots for
        chosen_parameter_numbers = plot_model['param_number']
        chosen_parameters = [plot_model['parameters'][n-1] for n in chosen_parameter_numbers]

        # Gathering data for the correlation calculations
        parameter_list = PatientDatabase.retrieve_parameters_from_all_patients(chosen_parameter_numbers, chosen_parameters)

        for module_number in plot_model['module_number']:
            module = plot_model['modules'][module_number-1]
            module = module_names[module]
            for tumour_number in plot_model['tumour_number']:
                tumour_type = plot_model['tumour type'][tumour_number-1]
                parameters = parameter_list[0][4*(tumour_number-1)+(module_number-1)]
                for i in range(0, len(chosen_parameters)-1):
                    chosen_parameter1 = plot_model['parameters'][plot_model['param_number'][i]-1]
                    chosen_parameter1_label = parameter_names[chosen_parameter1]
                    for j in range(i+1, len(chosen_parameters)):
                        chosen_parameter2 = plot_model['parameters'][plot_model['param_number'][j] - 1]
                        chosen_parameter2_label = parameter_names[chosen_parameter2]
                        parameter1 = parameters[:, i]
                        parameter2 = parameters[:, j]

                        xlabel = "{} {}".format(module, chosen_parameter1_label)
                        ylabel = "{} {}".format(module, chosen_parameter2_label)
                        PlotFunctions.CCC_estimation_and_plot(plot_dcm, parameter1, parameter2, xlabel, ylabel, False,
                                                              module, chosen_parameter1, chosen_parameter2,
                                                              tumour_type, patient_numbers['plots'])

    if 8 in plot_model['plot_number']:
        # Add the chosen patients to PatientDatabase
        PatientDatabase.add_patients(patient_numbers['plots'], AIF_slices, AIF_times)

        # Choose which parameters to compare and do CCC plots for
        chosen_parameter_numbers = plot_model['param_number']
        chosen_parameters = [plot_model['parameters'][n - 1] for n in chosen_parameter_numbers]

        # Gathering data for the correlation calculations
        parameter_list = PatientDatabase.retrieve_parameters_from_all_patients(chosen_parameter_numbers,
                                                                               chosen_parameters)

        for module_number in plot_model['module_number']:
            module = plot_model['modules'][module_number - 1]
            for tumour_number in plot_model['tumour_number']:
                tumour_type = plot_model['tumour type'][tumour_number - 1]
                parameters = parameter_list[0][4 * (tumour_number - 1) + (module_number - 1)]
                #correlation_path = os.path.join(calc_dcm, "Correlation/Correlation_{}_{}_p{}_"
                #                                          ".npz".format(module, tumour_type, [int(p)-1000 for p in patient_numbers['plots']]))
                correlation_path = os.path.join(calc_dcm, "Correlation_wo_outliers/CC_wo_outliers_{}_{}_.npz".format(module, tumour_type))


                correlation_data = np.load(correlation_path)
                pearson_CC_matrix = correlation_data['CC']
                pearson_pval_matrix = correlation_data['pvalue']

                """
                pearson_CC_matrix = correlation_data['Pearson']
                pearson_pval_matrix = correlation_data['Pearson_pval']
                spearman_CC_matrix = correlation_data['Spearman']
                spearman_pval_matrix = correlation_data['Spearman_pval']
                """

                for i in range(0, len(chosen_parameters) - 1):
                    chosen_parameter1 = plot_model['parameters'][plot_model['param_number'][i] - 1]
                    chosen_parameter1_label = parameter_names[chosen_parameter1]
                    for j in range(i + 1, len(chosen_parameters)):
                        chosen_parameter2 = plot_model['parameters'][plot_model['param_number'][j] - 1]
                        chosen_parameter2_label = parameter_names[chosen_parameter2]
                        parameter1 = parameters[:, i]
                        parameter2 = parameters[:, j]
                        pearson_CC = pearson_CC_matrix[plot_model['param_number'][i]-1][plot_model['param_number'][j]-1]
                        pearson_pval = pearson_pval_matrix[plot_model['param_number'][i]-1][plot_model['param_number'][j]-1]
                        #spearman_CC = spearman_CC_matrix[plot_model['param_number'][i]-1][plot_model['param_number'][j]-1]
                        #spearman_pval = spearman_pval_matrix[plot_model['param_number'][i]-1][plot_model['param_number'][j]-1]
                        spearman_CC = 0
                        spearman_pval = 0

                        xlabel = "{} {}".format(module, chosen_parameter1_label)
                        ylabel = "{} {}".format(module, chosen_parameter2_label)
                        PlotFunctions.Linear_regression_and_Pearson_CC(plot_dcm, parameter1, parameter2,
                                                                       pearson_CC, spearman_CC, pearson_pval, spearman_pval,
                                                                       xlabel, ylabel, module, chosen_parameter1,
                                                                       chosen_parameter2,
                                                                       tumour_type, patient_numbers['plots'])

    # Plot heatmaps
    if 9 in plot_model['plot_number']:
        # Add the chosen patients to PatientDatabase
        PatientDatabase.add_patients(patient_numbers['plots'], AIF_slices, AIF_times)

        # Choose which parameters to compare and do CCC plots for
        chosen_parameter_numbers = plot_model['param_number']
        chosen_parameters = [plot_model['parameters'][n - 1] for n in chosen_parameter_numbers]
        chosen_parameter_labels = [parameter_names[p] for p in chosen_parameters]

        # Gathering data for the correlation calculations
        #parameter_list = PatientDatabase.retrieve_parameters_from_all_patients(chosen_parameter_numbers,
        #                                                                       chosen_parameters)

        for module_number in plot_model['module_number']:
            module = plot_model['modules'][module_number - 1]
            for tumour_number in plot_model['tumour_number']:
                tumour_type = plot_model['tumour type'][tumour_number - 1]
                correlation_path = os.path.join(calc_dcm, "Correlation/Correlation_{}_{}_p{}_"
                                                          ".npz".format(module, tumour_type, [int(p)-1000 for p in patient_numbers['plots']]))
                correlation_data = np.load(correlation_path)
                pearson_CC_matrix = correlation_data['Pearson']
                pearson_pval_matrix = correlation_data['Pearson_pval']
                spearman_CC_matrix = correlation_data['Spearman']
                spearman_pval_matrix = correlation_data['Spearman_pval']

                PlotFunctions.heatmaps(plot_dcm, pearson_CC_matrix, chosen_parameter_labels, "Correlation coefficient",
                                       [-1, 1], "RdBu", module, tumour_type, patient_numbers, color_graded=True)
                PlotFunctions.heatmaps(plot_dcm, pearson_pval_matrix, chosen_parameter_labels,
                                       "P value", [0, 1], "Blues", module, tumour_type, patient_numbers, color_graded=True)

    if 10 in plot_model['plot_number']:
        for patient in patient_numbers['plots']:
            for model_number in plot_model['model_number']:
                model = plot_model['models'][model_number-1]


                tumour_segmentation_folder = os.path.join(data_dcm, "{}_EMIN_{}_EMIN/nifti/dce_segmented/Tumours".format(patient, patient))
                chosen_tumour_names = []
                for tumour_name in os.listdir(tumour_segmentation_folder):
                    if tumour_name[:4] == 'mask' and tumour_name[5:9] == "GTVn":
                        chosen_tumour_names.append(tumour_name[5:-7])

                for tumour_type in chosen_tumour_names:
                    plot_loc = os.path.join(plot_dcm, "Fitting comparison/{}_EMIN_{}_EMIN/{}".format(patient, patient, model))
                    if os.path.exists(plot_loc) == False:
                        os.makedirs(plot_loc)
                    if model == "Brix":
                        ROI_path = os.path.join(calc_dcm,
                                                "PatientDatabase/{}_EMIN_{}_EMIN/Parameters/{}/MeanROI_param_model_{}_tumour_{}_rm_bl_.npz".format(
                                                    patient, patient, model, model, tumour_type))
                        dicom_path = os.path.join(data_dcm, "{}_EMIN_{}_EMIN/dicom".format(patient, patient)) # To calculate tau
                        ROI_data = np.load(ROI_path)
                        signal = ROI_data["S_S0_mean"]
                        signal = np.mean(signal, axis=1)
                        PlotFunctions.compare_fittings(plot_loc, time, signal, model, tumour_type, dicom=dicom_path,
                                                       sek_or_min=sek_or_min, injection_speed=injection_speed)

                    else:
                        Old_ROI_path = os.path.join(calc_dcm,
                                                "PatientDatabase/{}_EMIN_{}_EMIN/Parameters/{}/MeanROI_param_IndAIF_s{}_model_{}_tumour_{}_.npz".format(
                                                    patient, patient, model, AIF_slices[patient], model, tumour_type))
                        ROI_path = os.path.join(calc_dcm,
                                                "PatientDatabase/{}_EMIN_{}_EMIN/Parameters/{}/MeanROI_param_IndAIF_t{}_model_{}_tumour_{}_.npz".format(
                                                    patient, patient, model, AIF_times[patient], model, tumour_type))
                        ROI_data = np.load(ROI_path)
                        Cp = ROI_data["Cp"]
                        Ct = ROI_data["Ct_mean"]
                        Ct = np.mean(Ct, axis=1)
                        PlotFunctions.compare_fittings(plot_loc, time, Ct, model, tumour_type, Cp=Cp)

    # Plot box plot of parameters for different TNM staging
    if 11 in plot_model['plot_number']:
        # Add the chosen patients to PatientDatabase
        PatientDatabase.add_patients(patient_numbers['plots'], AIF_slices, AIF_times)

        # Choose which parameters to do correlation with TNM
        chosen_parameter_numbers = plot_model['param_number']
        chosen_parameters = [plot_model['parameters'][n - 1] for n in chosen_parameter_numbers]


        ROI_params_index_dict = {'Kep_Brix': 0, 'Kel_Brix': 1, 'A_Brix': 2, 'Kt_TM': 3, 've_TM': 4, 'Kep_TM': 5,
                         'Kt_ETM': 6, 've_ETM': 7, 'vp_ETM': 8, 'Kep_ETM': 9, 'AUC60': 10, 'AUC90': 11, 'AUC120': 12,
                                 'TTHP': 13}

        TNM_folder_path = os.path.join(data_dcm, "TNM")
        TNM_path = os.path.join(TNM_folder_path, "TNM.csv")
        T_negative, T_positive, T_both, N_negative, N_positive, N_both = RaWCSV.get_TNM_data(TNM_path)
        TNM_data_dict = {'T_negative': T_negative, 'T_positive': T_positive, 'T_both': T_both,
                    'N_negative': N_negative, 'N_positive': N_positive, 'N_both': N_both}
        tumour_type = {'T_negative': 'p', 'T_positive': 'p', 'T_both': 'p',
                    'N_negative': 'n', 'N_positive': 'n', 'N_both': 'n'}

        colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:blue']

        axes_upper_limits = {'Kep_Brix': 12, 'Kel_Brix': 0.25, 'A_Brix': 140, 'Kt_TM': 3, 've_TM': 1, 'Kep_TM': 3,
                             'Kt_ETM': 3, 've_ETM': 1, 'vp_ETM': 0.4, 'Kep_ETM': 3, 'AUC60': 120, 'AUC90': 180,
                             'AUC120': 250, 'TTHP': 2.5}

        axes_lower_limits = {'Kep_Brix': 0, 'Kel_Brix': -0.05, 'A_Brix': 10, 'Kt_TM': 0, 've_TM': 0, 'Kep_TM': 0,
                             'Kt_ETM': 0, 've_ETM': 0, 'vp_ETM': 0, 'Kep_ETM': 0, 'AUC60': 20, 'AUC90': 40,
                             'AUC120': 70, 'TTHP': 0}


        for HPV_type_number in plot_model['HPV_type_number']:
            HPV_type = plot_model['HPV type'][HPV_type_number-1]
            TNM_data = TNM_data_dict[HPV_type]

            if HPV_type[0] == 'T':
                n_or_p = 'p'
                xlabel = 'T staging'
                S1 = [k for k,v in TNM_data.items() if float(v) == 1]
                S2 = [k for k,v in TNM_data.items() if float(v) == 2]
                S3 = [k for k,v in TNM_data.items() if float(v) == 3]
                S4 = [k for k,v in TNM_data.items() if float(v) == 4]


            else:
                n_or_p = 'n'
                xlabel = 'N staging'
                S1 = [k for k, v in TNM_data.items() if float(v) == 1]
                S2 = [k for k, v in TNM_data.items() if float(v) == 2]
                S3 = [k for k, v in TNM_data.items() if float(v) == 3]
                S4 = []

            # Get ROI parameters
            for parameter_metadata in chosen_parameters:
                if parameter_metadata[:3] != 'AUC' and parameter_metadata != 'TTHP':
                    parameter_metadata_list = parameter_metadata.split('_')
                    parameter = parameter_metadata_list[0]
                    model = parameter_metadata_list[1]


                else:
                    parameter = parameter_metadata
                    if parameter_metadata[:3] == 'AUC':
                        model = 'AUC'
                    else:
                        model = 'TTHP'

                S1_ROI_values = []
                S2_ROI_values = []
                S3_ROI_values = []

                S1_patients = []
                S2_patients = []
                S3_patients = []

                Low_ROI_values = []
                High_ROI_values = []

                if HPV_type[0] == 'T':
                    S4_ROI_values = []
                    S4_patients = []

                for Patient in PatientDatabase.chosen_patients:
                    patient_name = Patient.patient_name[:4]
                    Patient.add_tumours()
                    tumours = Patient.chosen_tumour_names
                    tumours = [tumour for tumour in tumours if tumour[3]==n_or_p]
                    for tumour in tumours:
                        if model == 'TM' or model == 'ETM':
                            AIF = "IndAIF"
                            param = Patient.retrieve_single_ROI_parameter(parameter, model, tumour, AIF=AIF)
                        else:
                            param = Patient.retrieve_single_ROI_parameter(parameter, model, tumour)
                        if patient_name in S1:
                            S1_ROI_values = np.append(S1_ROI_values, param)
                            Low_ROI_values = np.append(Low_ROI_values, param)
                            S1_patients = np.append(S1_patients, patient_name)

                        elif patient_name in S2:
                            S2_ROI_values = np.append(S2_ROI_values, param)
                            Low_ROI_values = np.append(Low_ROI_values, param)
                            S2_patients = np.append(S2_patients, patient_name)


                        elif patient_name in S3:
                            S3_ROI_values = np.append(S3_ROI_values, param)
                            High_ROI_values = np.append(High_ROI_values, param)
                            S3_patients = np.append(S3_patients, patient_name)


                        if HPV_type[0] == 'T' and patient_name in S4:
                            S4_ROI_values = np.append(S4_ROI_values, param)
                            High_ROI_values = np.append(High_ROI_values, param)
                            S4_patients = np.append(S4_patients, patient_name)


                if HPV_type[0] == 'T':
                    ROI_param_TN_list = [S1_ROI_values, S2_ROI_values, S3_ROI_values, S4_ROI_values]
                    #ROI_param_TN_list = [Low_ROI_values, High_ROI_values]
                else:
                    ROI_param_TN_list = [S1_ROI_values, S2_ROI_values, S3_ROI_values]

                PlotFunctions.TNM_boxplot(plot_dcm, ROI_param_TN_list, parameter, model,
                                          xlabel, parameter_names[parameter_metadata], HPV_type,
                                          axes_lower_limits[parameter_metadata], axes_upper_limits[parameter_metadata])
                RaWCSV.save_T_stage_data(ROI_param_TN_list, TNM_folder_path, parameter, HPV_type)



    # Plot comparison of parameters from IndAIF and PopAIF in one plot
    if 12 in plot_model['plot_number']:
        # Dictionary linking popAIF type to label
        popAIF_label_dict = {'PopAIF': 'W/ baseline',
                             'PopAIFrmBl': 'Wo/ baseline',
                             'PopAIFpa': 'W/ baseline and peak alignment',
                             'PopAIFpaRmBl': 'Wo/ baseline and peak alignment',
                             'PopAIFwia': 'W/ baseline and wash-in alignment',
                             'PopAIFwiaRmBl': 'Wo/ baseline and wash-in alignment'}

        color_dict = {'PopAIF': 'tab:blue',
                             'PopAIFrmBl': 'tab:blue',
                             'PopAIFpa': 'tab:orange',
                             'PopAIFpaRmBl': 'tab:orange',
                             'PopAIFwia': 'tab:olive',
                             'PopAIFwiaRmBl': 'tab:olive'}

        linestyle_dict = {'PopAIF': '--',
                             'PopAIFrmBl': '-.',
                             'PopAIFpa': '--',
                             'PopAIFpaRmBl': '-.',
                             'PopAIFwia': '--',
                             'PopAIFwiaRmBl': '-.'}

        # Add the chosen patients to PatientDatabase
        PatientDatabase.add_patients(patient_numbers['plots'], AIF_slices, AIF_times)

        # Choose which parameters to compare and do CCC plots for
        chosen_parameter_numbers = plot_model['param_number']
        chosen_parameters = [plot_model['parameters'][n - 1] for n in chosen_parameter_numbers]

        # Choose which population AIF to plot for
        chosen_popAIF_type_numbers = plot_model['PopAIF_type_number']
        popAIFs = [plot_model['PopAIF_type'][n - 1] for n in chosen_popAIF_type_numbers]

        # Choose whether you want to plot just nodal, primary or both tumour types
        chosen_tumour_type_number = plot_model['tumour_number']
        tumour_types = [plot_model['tumour type'][n - 1] for n in chosen_tumour_type_number]

        # Choose module (VbyV, ROI etc)
        chosen_module_number = plot_model['module_number']
        modules = [plot_model['modules'][n - 1] for n in chosen_module_number]

        # Chosen population AIFs
        chosen_popAIF_number = plot_model['PopAIF_number']
        popAIF_files = [plot_model['PopAIFs'][n - 1] for n in chosen_popAIF_number]

        for popAIF_file in popAIF_files:
            # The patients the population AIF is based on
            popAIF_metadata = popAIF_file.split('_')
            popAIF_patients = popAIF_metadata[1]

            for parameter in chosen_parameters:
                parameter_metadata = parameter.split('_')
                parameter_name = parameter_metadata[0]
                model = parameter_metadata[1]
                parameter_plot_name = parameter_names[parameter]
                for module in modules:
                    for tumour_type in tumour_types:
                        IndAIF_parameter_list = PatientDatabase.retrieve_IndAIF_and_PopAIF_parameters_from_patients(
                            parameter_name,
                            model, "IndAIF", popAIF_patients, tumour_type, module)

                        IndAIF_rm_bl_parameter_list = PatientDatabase.retrieve_IndAIF_and_PopAIF_parameters_from_patients(
                            parameter_name, model, "IndAIF_rm_bl", popAIF_patients, tumour_type, module)

                        All_PopAIF_parameter_list = {}
                        for popAIF_type in popAIFs:
                            # Gathering data for the correlation calculations
                            PopAIF_parameter_list = PatientDatabase.retrieve_IndAIF_and_PopAIF_parameters_from_patients(
                                parameter_name, model, popAIF_type, popAIF_patients, tumour_type, module)

                            All_PopAIF_parameter_list[popAIF_type] = PopAIF_parameter_list

                        PlotFunctions.CCC_estimation_and_plot_for_all_popAIF(plot_dcm, IndAIF_parameter_list,
                                      IndAIF_rm_bl_parameter_list, All_PopAIF_parameter_list, popAIFs, "Individual AIF",
                                      popAIF_label_dict, False, color_dict, linestyle_dict, module, parameter_name,
                                      parameter_plot_name, tumour_type)

    # Plot heatmaps of Pearson CC where outliers were removed
    if 13 in plot_model['plot_number']:
        parameters = ['Kep_Brix', 'Kel_Brix', 'A_Brix',
                      'Kt_TM', 've_TM','Kep_TM',
                      'Kt_ETM', 've_ETM', 'vp_ETM', 'Kep_ETM',
                      'AUC60', 'AUC90', 'AUC120',
                      'TTHP']

        parameter_names = {'Kep_Brix': r'K$_{ep, \ Brix}$ (min$^{-1}$)', 'Kel_Brix': r'K$_{el, \ Brix}$ (min$^{-1}$)',
                           'A_Brix': r'A',
                           'Kt_TM': r'K$^{trans}_{TM}$ (min$^{-1}$)', 've_TM': r'v$_{e,\ TM}$',
                           'Kep_TM': r'K$_{ep,\ TM}$ (min$^{-1}$)',
                           'Kt_ETM': r'K$^{trans}_{ETM}$ (min$^{-1}$)', 've_ETM': r'v$_{e,\ ETM}$',
                           'vp_ETM': r'v$_{p,\ ETM}$', 'Kep_ETM': r'K$_{ep, \ ETM}$ (min$^{-1}$)',
                           'AUC60': 'AUC60 (min)', 'AUC90': 'AUC90 (min)', 'AUC120': 'AUC120 (min)',
                           'TTHP': 'TTHP (min)'}

        parameter_labels = [parameter_names[parameter] for parameter in parameters]
        patients = patient_numbers['plots']
        patients_for_path = [patient_name[2:4] for patient_name in patients]

        """
        for module_number in plot_model['module_number']:
            module = plot_model['modules'][module_number - 1]
            for tumour_number in plot_model['tumour_number']:
                tumour_type = plot_model['tumour type'][tumour_number - 1]
                correlation_path = os.path.join(calc_dcm, "Correlation_wo_outliers/CC_wo_outliers_{}_{}_p{}"
                                                          ".npz".format(module, tumour_type, patients_for_path))
                correlation_data = np.load(correlation_path)
                pearson_CC_matrix = correlation_data['CC']
                pearson_pval_matrix = correlation_data['pvalue']

                PlotFunctions.heatmaps(plot_dcm, pearson_CC_matrix, parameter_names, "Correlation coefficient",
                                       [-1, 1], "RdBu", module, tumour_type, patient_numbers)
                PlotFunctions.heatmaps(plot_dcm, pearson_pval_matrix, parameter_labels,
                                       "P value", [0, 1], "Blues", module, tumour_type, patient_numbers)
        """
        for CC_number in plot_model['CC_number']:
            CC_matrix_file = plot_model['CC_matrices'][CC_number-1]
            CC_matrix_metadata = CC_matrix_file.split("_")
            module = CC_matrix_metadata[3]
            tumour_type = CC_matrix_metadata[4]
            patients = CC_matrix_metadata[5]

            CC_dir = os.path.join(calc_dcm, "Correlation_wo_outliers")
            CC_matrix_path = os.path.join(CC_dir, CC_matrix_file)

            correlation_data = np.load(CC_matrix_path)
            pearson_CC_matrix = correlation_data['CC']
            pearson_pval_matrix = correlation_data['pvalue']

            PlotFunctions.heatmaps(plot_dcm, pearson_CC_matrix, parameter_labels, "Pearson correlation coefficient",
                                   "CC", [-1, 1], "RdBu", module, tumour_type, patients, color_graded=True)
            PlotFunctions.heatmaps(plot_dcm, pearson_pval_matrix, parameter_labels, "Pearson p-value", "Pvalue",
                                   [0, 0.05], "Blues", module, tumour_type, patients, color_graded=True)

    if 14 in plot_model['plot_number']:
        # Choose which parameters to compare and do CCC plots for
        chosen_parameter_numbers = plot_model['param_number']
        chosen_parameters = [plot_model['parameters'][n - 1] for n in chosen_parameter_numbers]

        CC_dir = os.path.join(calc_dcm, "Correlation_wo_outliers")
        for CC_number in plot_model['CC_number']:
            CC_file = plot_model['CC_matrices'][CC_number - 1]
            CC_metadata = CC_file.split("_")
            module = CC_metadata[3]
            tumour_type = CC_metadata[4]
            patient_type = CC_metadata[5]
            CC_path = os.path.join(CC_dir, CC_file)
            correlation_data = np.load(CC_path)
            patients = correlation_data['patients']

            # Initialise patient database
            PatientDatabase = Database.PatientDatabase(data_dcm, calc_dcm, time, sek_or_min, T1, R_Gd, TR, FA,
                                                       injection_time, injection_speed, TA, dt)

            # Available patients
            available_patients = PatientDatabase.available_patients

            # Add patients used in the correlation matrix
            PatientDatabase.add_patients(patients, AIF_slices, AIF_times)

            pearson_CC_matrix = correlation_data['CC']
            pearson_pval_matrix = correlation_data['pvalue']

            for i in range(0, len(chosen_parameters) - 1):
                chosen_parameter1_metadata = plot_model['parameters'][plot_model['param_number'][i] - 1]
                chosen_parameter1_metadata_list = chosen_parameter1_metadata.split("_")
                chosen_parameter1 = chosen_parameter1_metadata_list[0]
                try:
                    chosen_model1 = chosen_parameter1_metadata_list[1]
                except:
                    if chosen_parameter1_metadata == 'TTHP':
                        chosen_model1 = 'TTHP'
                    else:
                        chosen_model1 = "AUC"
                chosen_parameter1_label = parameter_names[chosen_parameter1_metadata]

                for j in range(i + 1, len(chosen_parameters)):
                    chosen_parameter2_metadata = plot_model['parameters'][plot_model['param_number'][j] - 1]
                    chosen_parameter2_metadata_list = chosen_parameter2_metadata.split("_")
                    chosen_parameter2 = chosen_parameter2_metadata_list[0]
                    try:
                        chosen_model2 = chosen_parameter2_metadata_list[1]
                    except:
                        if chosen_parameter2_metadata == 'TTHP':
                            chosen_model2 = 'TTHP'
                        else:
                            chosen_model2 = "AUC"

                    chosen_parameter2_label = parameter_names[chosen_parameter2_metadata]

                    parameter1_list = np.array([])
                    parameter2_list = np.array([])
                    for Patient in PatientDatabase.chosen_patients:
                        Patient.add_tumours()
                        for tumour in Patient.chosen_tumour_names:
                            if tumour_type == 'nANDp' or tumour[3] == tumour_type[0]:
                                parameter1 = Patient.retrieve_single_ROI_parameter(chosen_parameter1, chosen_model1, tumour, AIF="IndAIF")
                                parameter2 = Patient.retrieve_single_ROI_parameter(chosen_parameter2, chosen_model2, tumour, AIF="IndAIF")

                                parameter1_list = np.append(parameter1_list, parameter1)
                                parameter2_list = np.append(parameter2_list, parameter2)

                    parameter1_list, parameter2_list, outliers = Modelling.remove_outliers(parameter1_list,
                                                                                           parameter2_list)

                    pearson_CC = pearson_CC_matrix[plot_model['param_number'][i]-1][plot_model['param_number'][j]-1]
                    pearson_pval = pearson_pval_matrix[plot_model['param_number'][i]-1][plot_model['param_number'][j]-1]

                    xlabel = "{} {}".format(module, chosen_parameter1_label)
                    ylabel = "{} {}".format(module, chosen_parameter2_label)
                    PlotFunctions.Linear_regression_and_Pearson_CC(plot_dcm, parameter1_list, parameter2_list,
                                                                   pearson_CC, 0, pearson_pval, 0,
                                                                   xlabel, ylabel, module, chosen_parameter1,
                                                                   chosen_parameter2,
                                                                   tumour_type, patients, patient_type)

    if 15 in plot_model['plot_number']:
        # Choose which parameters to compare and do CCC plots for
        chosen_parameter_numbers = plot_model['param_number']
        chosen_parameters = [plot_model['parameters'][n - 1] for n in chosen_parameter_numbers]

        CC_dir = os.path.join(calc_dcm, "Correlation_wo_outliers")
        for CC_number in plot_model['CC_number']:
            CC_file = plot_model['CC_matrices'][CC_number-1]
            CC_metadata = CC_file.split("_")
            module = CC_metadata[3]
            tumour_type = CC_metadata[4]
            patient_type = CC_metadata[5]
            CC_path = os.path.join(CC_dir, CC_file)
            correlation_data = np.load(CC_path)
            patients = correlation_data['patients']

            # Initialise patient database
            PatientDatabase = Database.PatientDatabase(data_dcm, calc_dcm, time, sek_or_min, T1, R_Gd, TR, FA,
                                                       injection_time, injection_speed, TA, dt)

            # Available patients
            available_patients = PatientDatabase.available_patients

            # Add patients used in the correlation matrix
            PatientDatabase.add_patients(patients, AIF_slices, AIF_times)

            pearson_CC_matrix = correlation_data['CC']
            pearson_pval_matrix = correlation_data['pvalue']

            for i in range(0, len(chosen_parameters) - 1):
                chosen_parameter1_metadata = plot_model['parameters'][plot_model['param_number'][i] - 1]
                chosen_parameter1_metadata_list = chosen_parameter1_metadata.split("_")
                chosen_parameter1 = chosen_parameter1_metadata_list[0]
                try:
                    chosen_model1 = chosen_parameter1_metadata_list[1]
                except:
                    chosen_model1 = "AUC"
                chosen_parameter1_label = parameter_names[chosen_parameter1_metadata]

                for j in range(i + 1, len(chosen_parameters)):
                    chosen_parameter2_metadata = plot_model['parameters'][plot_model['param_number'][j] - 1]
                    chosen_parameter2_metadata_list = chosen_parameter2_metadata.split("_")
                    chosen_parameter2 = chosen_parameter2_metadata_list[0]
                    try:
                        chosen_model2 = chosen_parameter2_metadata_list[1]
                    except:
                        chosen_model2 = "AUC"
                    chosen_parameter2_label = parameter_names[chosen_parameter2_metadata]

                    parameter1_list = np.array([])
                    parameter2_list = np.array([])
                    for Patient in PatientDatabase.chosen_patients:
                        Patient.add_tumours()
                        for tumour in Patient.chosen_tumour_names:
                            if tumour_type == 'nANDp' or tumour[3] == tumour_type[0]:
                                parameter1 = Patient.retrieve_single_ROI_parameter(chosen_parameter1, chosen_model1, tumour, AIF="IndAIF")
                                parameter2 = Patient.retrieve_single_ROI_parameter(chosen_parameter2, chosen_model2, tumour, AIF="IndAIF")

                                parameter1_list = np.append(parameter1_list, parameter1)
                                parameter2_list = np.append(parameter2_list, parameter2)

                    parameter1_list, parameter2_list, outliers = Modelling.remove_outliers(parameter1_list,
                                                                                           parameter2_list)

                    pearson_CC = pearson_CC_matrix[plot_model['param_number'][i]-1][plot_model['param_number'][j]-1]
                    pearson_pval = pearson_pval_matrix[plot_model['param_number'][i]-1][plot_model['param_number'][j]-1]

                    xlabel = "{} {}".format(module, chosen_parameter1_label)
                    ylabel = "{} {}".format(module, chosen_parameter2_label)

                    PlotFunctions.CCC_estimation_and_plot(plot_dcm, parameter1_list, parameter2_list, xlabel, ylabel, False,
                                                          module, chosen_parameter1, chosen_parameter2, chosen_model1,
                                                          chosen_model2, tumour_type, patients, patient_type)

    # Plot three heatmaps of Pearson CC where outliers were removed
    if 16 in plot_model['plot_number']:
        parameters = ['Kep_Brix', 'Kel_Brix', 'A_Brix',
                      'Kt_TM', 've_TM', 'Kep_TM',
                      'Kt_ETM', 've_ETM', 'vp_ETM', 'Kep_ETM',
                      'AUC60', 'AUC90', 'AUC120',
                      'TTHP']

        parameter_names = {'Kep_Brix': r'K$_{ep, \ Brix}$ (min$^{-1}$)', 'Kel_Brix': r'K$_{el, \ Brix}$ (min$^{-1}$)',
                           'A_Brix': r'A',
                           'Kt_TM': r'K$^{trans}_{TM}$ (min$^{-1}$)', 've_TM': r'v$_{e,\ TM}$',
                           'Kep_TM': r'K$_{ep,\ TM}$ (min$^{-1}$)',
                           'Kt_ETM': r'K$^{trans}_{ETM}$ (min$^{-1}$)', 've_ETM': r'v$_{e,\ ETM}$',
                           'vp_ETM': r'v$_{p,\ ETM}$', 'Kep_ETM': r'K$_{ep, \ ETM}$ (min$^{-1}$)',
                           'AUC60': 'AUC60 (min)', 'AUC90': 'AUC90 (min)', 'AUC120': 'AUC120 (min)',
                           'TTHP': 'TTHP (min)'}

        parameter_labels = [parameter_names[parameter] for parameter in parameters]
        patients = patient_numbers['plots']
        patients_for_path = [patient_name[2:4] for patient_name in patients]

        matrix_types = ['CC', 'pvalue']
        colorbar_label = ['Pearson correlation coefficient', 'Pearson p-value']
        colorbar_min = [-1, 0]
        colorbar_max = [1, 0.05]
        colorbar_cmap = ['RdBu', 'Blues']
        for j, matrix_type in enumerate(matrix_types):

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 12))
            for i, CC_number in enumerate(plot_model['CC_number']):
                CC_matrix_file = plot_model['CC_matrices'][CC_number - 1]
                CC_matrix_metadata = CC_matrix_file.split("_")
                module = CC_matrix_metadata[3]
                tumour_type = CC_matrix_metadata[4]
                patients = CC_matrix_metadata[5]

                CC_dir = os.path.join(calc_dcm, "Correlation_wo_outliers")
                CC_matrix_path = os.path.join(CC_dir, CC_matrix_file)

                correlation_data = np.load(CC_matrix_path)
                pearson_CC_matrix = correlation_data[matrix_type]

                heatmap_dir = os.path.join(plot_dcm, "Heatmaps_wo_outliers/{}".format(module))
                if os.path.exists(heatmap_dir) == False:
                    os.makedirs(heatmap_dir)

                heatmap_path = os.path.join(heatmap_dir,
                                            "{}_heatmap_{}_{}_full_figure.png".format(matrix_type, module, patients))

                params_orig = {"Kep_Brix": 0, "Kel_Brix": 1, "A": 2, "Kt_TM": 3, "ve_TM": 4, "Kep_TM": 5, "Kt_ETM": 6,
                                   "ve_ETM": 7, "vp_ETM": 8, "Kep_ETM": 9, "AUC60": 10, "AUC90": 11, "AUC120": 12,
                                   "TTHP": 13}
                sorted_params = ["AUC60", "AUC90", "AUC120", "A", "Kt_ETM", "Kt_TM", "ve_ETM", "ve_TM", "Kep_ETM",
                                     "Kep_TM",
                                     "Kep_Brix", "Kel_Brix", "vp_ETM", "TTHP"]
                sorted_params_indx = [params_orig[key] for key in sorted_params]
                rows = np.array([np.ones(14, dtype=int) * sorted_params_indx[i] for i in range(14)])
                matrix = pearson_CC_matrix[rows, rows.T]
                features = np.array(parameter_labels)[sorted_params_indx]

                subimg = ax[i].imshow(matrix, cmap=colorbar_cmap[j], vmin=colorbar_min[j], vmax=colorbar_max[j])

                ticks_list = np.arange(0, len(features), 1)
                ax[i].set_xticks(ticks_list, features, rotation='vertical')
                ax[i].tick_params(axis='both', which='major', labelsize=26)
                ax[i].tick_params(axis='both', which='minor', labelsize=26)

                if i==2:
                    cax = fig.add_axes([0.9, 0.253, 0.02, 0.48])
                    cbar = fig.colorbar(mappable=subimg, cax=cax)
                    cbar.ax.tick_params(labelsize=26)
                    cbar.set_label(colorbar_label[j], size=32, labelpad=15)

            ax[0].set_yticks(ticks_list, features)
            ax[1].tick_params(left=False, labelleft=False)
            ax[2].tick_params(left=False, labelleft=False)

            left, width = 0, 1
            bottom, height = 0, 1
            right = left + width
            top = bottom + height

            p = plt.Rectangle((left, bottom), width, height, fill=False)
            p.set_transform(ax[0].transAxes)
            p.set_clip_on(False)
            ax[0].add_patch(p)
            ax[0].text(0.5*(left+right), top+0.05, "(a)", fontsize=34, weight='bold', horizontalalignment='center',
                       verticalalignment='bottom', transform=ax[0].transAxes)

            p = plt.Rectangle((left, bottom), width, height, fill=False)
            p.set_transform(ax[1].transAxes)
            p.set_clip_on(False)
            ax[1].add_patch(p)
            ax[1].text(0.5*(left+right), top+0.05, "(b)", fontsize=34, weight='bold', horizontalalignment='center',
                       verticalalignment='bottom', transform=ax[1].transAxes)

            p = plt.Rectangle((left, bottom), width, height, fill=False)
            p.set_transform(ax[2].transAxes)
            p.set_clip_on(False)
            ax[2].add_patch(p)
            ax[2].text(0.5*(left+right), top+0.05, "(c)", fontsize=34, weight='bold', horizontalalignment='center',
                       verticalalignment='bottom', transform=ax[2].transAxes)

            plt.subplots_adjust(right=0.88, wspace=0.1)

            plt.savefig(heatmap_path)
            plt.show()

    # Plot four (2x2) heatmaps of Pearson CC where outliers were removed
    if 17 in plot_model['plot_number']:
        parameters = ['Kep_Brix', 'Kel_Brix', 'A_Brix',
                      'Kt_TM', 've_TM', 'Kep_TM',
                      'Kt_ETM', 've_ETM', 'vp_ETM', 'Kep_ETM',
                      'AUC60', 'AUC90', 'AUC120',
                      'TTHP']

        parameter_names = {'Kep_Brix': r'K$_{ep, \ Brix}$ (min$^{-1}$)', 'Kel_Brix': r'K$_{el, \ Brix}$ (min$^{-1}$)',
                           'A_Brix': r'A',
                           'Kt_TM': r'K$^{trans}_{TM}$ (min$^{-1}$)', 've_TM': r'v$_{e,\ TM}$',
                           'Kep_TM': r'K$_{ep,\ TM}$ (min$^{-1}$)',
                           'Kt_ETM': r'K$^{trans}_{ETM}$ (min$^{-1}$)', 've_ETM': r'v$_{e,\ ETM}$',
                           'vp_ETM': r'v$_{p,\ ETM}$', 'Kep_ETM': r'K$_{ep, \ ETM}$ (min$^{-1}$)',
                           'AUC60': 'AUC60 (min)', 'AUC90': 'AUC90 (min)', 'AUC120': 'AUC120 (min)',
                           'TTHP': 'TTHP (min)'}

        parameter_labels = [parameter_names[parameter] for parameter in parameters]
        patients = patient_numbers['plots']
        patients_for_path = [patient_name[2:4] for patient_name in patients]

        matrix_types = ['CC', 'pvalue']
        colorbar_label = ['Pearson correlation coefficient', 'Pearson p-value']
        colorbar_min = [-1, 0]
        colorbar_max = [1, 0.05]
        colorbar_cmap = ['RdBu', 'Blues']
        for j, matrix_type in enumerate(matrix_types):

            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(25, 20))
            for i, CC_number in enumerate(plot_model['CC_number']):
                CC_matrix_file = plot_model['CC_matrices'][CC_number - 1]
                CC_matrix_metadata = CC_matrix_file.split("_")
                module = CC_matrix_metadata[3]
                tumour_type = CC_matrix_metadata[4]
                patients = CC_matrix_metadata[5]

                CC_dir = os.path.join(calc_dcm, "Correlation_wo_outliers")
                CC_matrix_path = os.path.join(CC_dir, CC_matrix_file)
                correlation_data = np.load(CC_matrix_path)
                pearson_CC_matrix = correlation_data[matrix_type]


                heatmap_dir = os.path.join(plot_dcm, "Heatmaps_wo_outliers/{}".format(module))
                if os.path.exists(heatmap_dir) == False:
                    os.makedirs(heatmap_dir)

                heatmap_path = os.path.join(heatmap_dir,
                                            "{}_heatmap_{}_HPV_pos_vs_neg_full_figure.png".format(matrix_type, module))

                params_orig = {"Kep_Brix": 0, "Kel_Brix": 1, "A": 2, "Kt_TM": 3, "ve_TM": 4, "Kep_TM": 5, "Kt_ETM": 6,
                                   "ve_ETM": 7, "vp_ETM": 8, "Kep_ETM": 9, "AUC60": 10, "AUC90": 11, "AUC120": 12,
                                   "TTHP": 13}
                sorted_params = ["AUC60", "AUC90", "AUC120", "A", "Kt_ETM", "Kt_TM", "ve_ETM", "ve_TM", "Kep_ETM",
                                     "Kep_TM",
                                     "Kep_Brix", "Kel_Brix", "vp_ETM", "TTHP"]
                sorted_params_indx = [params_orig[key] for key in sorted_params]
                rows = np.array([np.ones(14, dtype=int) * sorted_params_indx[i] for i in range(14)])
                matrix = pearson_CC_matrix[rows, rows.T]
                features = np.array(parameter_labels)[sorted_params_indx]

                if i == 0:
                    ax = axs[0, 0]
                elif i == 1:
                    ax = axs[0, 1]
                elif i == 2:
                    ax = axs[1, 0]
                else:
                    ax = axs[1, 1]

                subimg = ax.imshow(matrix, cmap=colorbar_cmap[j], vmin=colorbar_min[j], vmax=colorbar_max[j])

                ticks_list = np.arange(0, len(features), 1)
                ax.tick_params(axis='both', which='major', labelsize=26)
                ax.tick_params(axis='both', which='minor', labelsize=26)

                if i==2:
                    cax = fig.add_axes([0.85, 0.15, 0.02, 0.8])
                    cbar = fig.colorbar(mappable=subimg, cax=cax)
                    cbar.ax.tick_params(labelsize=26)
                    cbar.set_label(colorbar_label[j], size=32, labelpad=15)

            axs[0, 0].set_yticks(ticks_list, features)
            axs[0, 0].tick_params(bottom=False, labelbottom=False)

            axs[0, 1].tick_params(left=False, labelleft=False)
            axs[0, 1].tick_params(bottom=False, labelbottom=False)

            axs[1, 0].set_yticks(ticks_list, features)
            axs[1, 0].set_xticks(ticks_list, features, rotation='vertical')

            axs[1, 1].tick_params(left=False, labelleft=False)
            axs[1, 1].set_xticks(ticks_list, features, rotation='vertical')

            left, width = 0, 1
            bottom, height = 0, 1
            right = left + width
            top = bottom + height

            p = plt.Rectangle((left, bottom), width, height, fill=False)
            p.set_transform(axs[0,0].transAxes)
            p.set_clip_on(False)
            axs[0,0].add_patch(p)
            axs[0,0].text(0.5*(left+right), top+0.05, "HPV positive", fontsize=34, weight='bold', horizontalalignment='center',
                       verticalalignment='bottom', transform=axs[0,0].transAxes)

            p = plt.Rectangle((left, bottom), width, height, fill=False)
            p.set_transform(axs[0, 1].transAxes)
            p.set_clip_on(False)
            axs[0, 1].add_patch(p)
            axs[0, 1].text(0.5*(left+right), top+0.05, "HPV negative", fontsize=34, weight='bold', horizontalalignment='center',
                       verticalalignment='bottom', transform=axs[0, 1].transAxes)

            p = plt.Rectangle((left, bottom), width, height, fill=False)
            p.set_transform(axs[0,0].transAxes)
            p.set_clip_on(False)
            axs[0,0].add_patch(p)
            axs[0,0].text(left-0.45, 0.5*(bottom+top), "Primary tumours", fontsize=34, weight='bold', rotation='vertical',
                          horizontalalignment='right', verticalalignment='center', transform=axs[0,0].transAxes)

            p = plt.Rectangle((left, bottom), width, height, fill=False)
            p.set_transform(axs[1,0].transAxes)
            p.set_clip_on(False)
            axs[1,0].add_patch(p)
            axs[1,0].text(left-0.45, 0.5 * (bottom + top), "Lymph nodes", fontsize=34, weight='bold', rotation='vertical',
                          horizontalalignment='right', verticalalignment='center', transform=axs[1,0].transAxes)

            plt.tight_layout()
            plt.subplots_adjust(right=0.88, wspace=-0.24, hspace=0.09)
            plt.savefig(heatmap_path)
            plt.show()

    if 18 in plot_model['plot_number']:
        # Add the chosen patients to PatientDatabase
        PatientDatabase.add_patients(patient_numbers['plots'], AIF_slices, AIF_times)

        # Choose which parameters to do correlation with TNM
        chosen_parameter_numbers = plot_model['param_number']
        chosen_parameters = [plot_model['parameters'][n - 1] for n in chosen_parameter_numbers]

        ROI_params_index_dict = {'Kep_Brix': 0, 'Kel_Brix': 1, 'A_Brix': 2, 'Kt_TM': 3, 've_TM': 4, 'Kep_TM': 5,
                                 'Kt_ETM': 6, 've_ETM': 7, 'vp_ETM': 8, 'Kep_ETM': 9, 'AUC60': 10, 'AUC90': 11,
                                 'AUC120': 12,
                                 'TTHP': 13}

        TNM_folder_path = os.path.join(data_dcm, "TNM")
        TNM_path = os.path.join(TNM_folder_path, "TNM.csv")
        T_negative, T_positive, T_both, N_negative, N_positive, N_both = RaWCSV.get_TNM_data(TNM_path)
        TNM_data_dict = {'T_negative': T_negative, 'T_positive': T_positive, 'T_both': T_both,
                         'N_negative': N_negative, 'N_positive': N_positive, 'N_both': N_both}
        tumour_type = {'T_negative': 'p', 'T_positive': 'p', 'T_both': 'p',
                       'N_negative': 'n', 'N_positive': 'n', 'N_both': 'n'}

        colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:blue']

        axes_upper_limits = {'Kep_Brix': 12, 'Kel_Brix': 0.25, 'A_Brix': 140, 'Kt_TM': 3, 've_TM': 1, 'Kep_TM': 4,
                             'Kt_ETM': 3, 've_ETM': 1, 'vp_ETM': 0.4, 'Kep_ETM': 3, 'AUC60': 120, 'AUC90': 180,
                             'AUC120': 250, 'TTHP': 2.5}

        axes_lower_limits = {'Kep_Brix': 0, 'Kel_Brix': -0.05, 'A_Brix': 10, 'Kt_TM': 0, 've_TM': 0, 'Kep_TM': 0,
                             'Kt_ETM': 0, 've_ETM': 0, 'vp_ETM': 0, 'Kep_ETM': 0, 'AUC60': 20, 'AUC90': 40,
                             'AUC120': 70, 'TTHP': 0}

        ROI_param_TN_complete_list = []
        for HPV_type_number in plot_model['HPV_type_number']:
            HPV_type = plot_model['HPV type'][HPV_type_number - 1]
            TNM_data = TNM_data_dict[HPV_type]

            if HPV_type[0] == 'T':
                n_or_p = 'p'
                xlabel = 'T staging'
                S1 = [k for k, v in TNM_data.items() if float(v) == 1]
                S2 = [k for k, v in TNM_data.items() if float(v) == 2]
                S3 = [k for k, v in TNM_data.items() if float(v) == 3]
                S4 = [k for k, v in TNM_data.items() if float(v) == 4]


            else:
                n_or_p = 'n'
                xlabel = 'N staging'
                S1 = [k for k, v in TNM_data.items() if float(v) == 1]
                S2 = [k for k, v in TNM_data.items() if float(v) == 2]
                S3 = [k for k, v in TNM_data.items() if float(v) == 3]
                S4 = []

            # Get ROI parameters
            for parameter_metadata in chosen_parameters:
                if parameter_metadata[:3] != 'AUC' and parameter_metadata != 'TTHP':
                    parameter_metadata_list = parameter_metadata.split('_')
                    parameter = parameter_metadata_list[0]
                    model = parameter_metadata_list[1]


                else:
                    parameter = parameter_metadata
                    if parameter_metadata[:3] == 'AUC':
                        model = 'AUC'
                    else:
                        model = 'TTHP'

                S1_ROI_values = []
                S2_ROI_values = []
                S3_ROI_values = []

                S1_patients = []
                S2_patients = []
                S3_patients = []

                Low_ROI_values = []
                High_ROI_values = []

                if HPV_type[0] == 'T':
                    S4_ROI_values = []
                    S4_patients = []

                for Patient in PatientDatabase.chosen_patients:
                    patient_name = Patient.patient_name[:4]
                    Patient.add_tumours()
                    tumours = Patient.chosen_tumour_names
                    tumours = [tumour for tumour in tumours if tumour[3] == n_or_p]
                    for tumour in tumours:
                        if model == 'TM' or model == 'ETM':
                            AIF = "IndAIF"
                            param = Patient.retrieve_single_ROI_parameter(parameter, model, tumour, AIF=AIF)
                        else:
                            param = Patient.retrieve_single_ROI_parameter(parameter, model, tumour)
                        if patient_name in S1:
                            S1_ROI_values = np.append(S1_ROI_values, param)
                            Low_ROI_values = np.append(Low_ROI_values, param)
                            S1_patients = np.append(S1_patients, patient_name)

                        elif patient_name in S2:
                            S2_ROI_values = np.append(S2_ROI_values, param)
                            Low_ROI_values = np.append(Low_ROI_values, param)
                            S2_patients = np.append(S2_patients, patient_name)


                        elif patient_name in S3:
                            S3_ROI_values = np.append(S3_ROI_values, param)
                            High_ROI_values = np.append(High_ROI_values, param)
                            S3_patients = np.append(S3_patients, patient_name)

                        if HPV_type[0] == 'T' and patient_name in S4:
                            S4_ROI_values = np.append(S4_ROI_values, param)
                            High_ROI_values = np.append(High_ROI_values, param)
                            S4_patients = np.append(S4_patients, patient_name)

                if HPV_type[0] == 'T':
                    ROI_param_TN_list = [S1_ROI_values, S2_ROI_values, S3_ROI_values, S4_ROI_values]
                    # ROI_param_TN_list = [Low_ROI_values, High_ROI_values]
                else:
                    ROI_param_TN_list = [S1_ROI_values, S2_ROI_values, S3_ROI_values]

                if ROI_param_TN_complete_list == []:
                    ROI_param_TN_complete_list = [ROI_param_TN_list]
                else:
                    ROI_param_TN_complete_list.append(ROI_param_TN_list)

                RaWCSV.save_T_stage_data(ROI_param_TN_list, TNM_folder_path, parameter_metadata, HPV_type)
        HPV_types = [plot_model['HPV type'][HPV_type_number - 1] for HPV_type_number in plot_model['HPV_type_number']]
        PlotFunctions.multiple_TNM_boxplot(ROI_param_TN_complete_list, HPV_types, chosen_parameters,
                                           'T staging', parameter_names, axes_lower_limits, axes_upper_limits, plot_dcm)

    if 19 in plot_model['plot_number']:
        # Limits for histogram
        axes_upper_limits = {'Kep_Brix': 12, 'Kel_Brix': 0.25, 'A_Brix': 140, 'Kt_TM': 3, 've_TM': 1, 'Kep_TM': 3,
                             'Kt_ETM': 3, 've_ETM': 1, 'vp_ETM': 0.4, 'Kep_ETM': 3, 'AUC60': 120, 'AUC90': 180,
                             'AUC120': 250, 'TTHP': 2.5}

        axes_lower_limits = {'Kep_Brix': 0, 'Kel_Brix': -0.05, 'A_Brix': 10, 'Kt_TM': 0, 've_TM': 0, 'Kep_TM': 0,
                             'Kt_ETM': 0, 've_ETM': 0, 'vp_ETM': 0, 'Kep_ETM': 0, 'AUC60': 20, 'AUC90': 40,
                             'AUC120': 70, 'TTHP': 0}

        # Add the chosen patients to PatientDatabase
        PatientDatabase.add_patients(patient_numbers['plots'], AIF_slices, AIF_times)

        for Patient in PatientDatabase.chosen_patients:
            Patient.add_tumours()
            patient_database_path = os.path.join(Patient.patient_database_path, Patient.patient_name)
            AIF_time = AIF_times[Patient.patient_number]
            for tumour, tumour_seg in zip(Patient.chosen_tumour_names, Patient.chosen_tumour_segmentations):
                for param_number in plot_model['param_number']:
                    parameter_metadata = plot_model['parameters'][param_number-1]
                    parameter_metadata_list = parameter_metadata.split("_")
                    parameter = parameter_metadata_list[0]
                    if parameter_metadata[:3] == "AUC":
                        model = "AUC"
                        param_folder = os.path.join(patient_database_path, "AUC")
                    else:
                        model = parameter_metadata_list[1]
                        param_folder = os.path.join(patient_database_path, "Parameters/{}".format(model))

                    if parameter[:3] == "AUC":
                        param_map_path = os.path.join(param_folder, "{}_tumour_{}_.nii.gz".format(parameter, tumour))
                    elif model == "Brix":
                        param_map_path = os.path.join(param_folder, "{}_model_{}_tumour_{}_.nii.gz".format(parameter, model, tumour))
                    else:
                        param_map_path = os.path.join(param_folder, "{}_IndAIF_t{}_model_{}_tumour_{}_.nii.gz".format(parameter, AIF_time, model, tumour))


                    PlotFunctions.parameter_histogram(param_map_path, tumour_seg, plot_dcm, Patient.patient_number,
                                                      parameter, model, tumour, axes_lower_limits[parameter_metadata],
                                                      axes_upper_limits[parameter_metadata])


main()

