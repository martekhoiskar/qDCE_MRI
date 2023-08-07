################################################################################
######################## I M P O R T  P A C K A G E S ##########################
################################################################################
import numpy as np
import os

def choose_quantity(calc_dcm):
    """
    Function that lets the user choose which quantities to calculate. For some quantities models, AIFs and methods
    to calculate correlation parameters must also be chosen.
    :return: a dictionary with the chosen quantities, models, AIFs and correlation methods
    """
    quant_model = {'quantities': ["Ct_tumour", "Individual AIF", "Population AIF",
                                  "Pharmacokinetic parameters", "AUC", "Correlation", "ICC for PopAIFs",
                                  "Correlation without outliers", "TTHP", "Export all"],

                   'models': ["TM", "ETM", "Brix", "Export all"],

                   'AIFs': ["Individual AIF", "Population AIF", "Export all"],

                   'modules': ["ROI", "All"],

                   'tumour': ['primary', 'nodal', 'nANDp', "All"]
    }

    # Print the different quantity options in the console
    for i in range(len(quant_model['quantities'])):
        print(str(i+1)+":\t"+quant_model['quantities'][i])

    # User chooses which quantities to calculate
    invalid_input=True
    while invalid_input:
        chosen_quantities = input("What quantitites do you want to calculate (seperated by <space>)? \n>>> ")
        quant_model['quantity_number'] = chosen_quantities.split()
        quant_model['quantity_number'] = [int(a) for a in quant_model['quantity_number']]
        if quant_model['quantity_number'] == [len(quant_model['quantities'])]:
            quant_model['quantity_number'] = np.arange(1, len(quant_model['quantities'])+1, 1, dtype=int)
        if all(q < (len(quant_model['quantities'])) for q in quant_model['quantity_number']):
            invalid_input=False
        else:
            print("One or more numbers were out of range. Choose again.")
            return

    # If pharmacokinetic parameters are to be calculated model(s) must be chosen
    if 4 in quant_model['quantity_number']:
        invalid_input = True
        while invalid_input:
            for i in range(len(quant_model['models'])):
                print(str(i+1)+":\t"+quant_model['models'][i])
            chosen_models = input("Which models do you want to use to calculate "
                                  "the pharmokinetic parameters (seperated by <space>)? \n>>> ")
            quant_model['model_number'] = chosen_models.split()
            quant_model['model_number'] = [int(a) for a in quant_model['model_number']]
            if quant_model['model_number'] == [len(quant_model['models'])]:
                quant_model['model_number'] = np.arange(1, len(quant_model['models']), 1, dtype=int)
            if all(q < (len(quant_model['models'])) for q in quant_model['model_number']):
                invalid_input=False
            else:
                print("One or more numbers were out of range. Choose again.")

    if 4 in quant_model['quantity_number']:
        # If the Tofts models are chosen, an AIF is necessary. User need to choose individual or population AIF.
        invalid_input = True
        while invalid_input:
            if 1 in quant_model['model_number'] or 2 in quant_model['model_number']:
                for i in range(len(quant_model['AIFs'])):
                    print(str(i+1)+":\t"+quant_model['AIFs'][i])
                chosen_AIF = input("Which AIF do you want to use for the pharmokinetic parameters? \n>>> ")
                quant_model['AIF_number'] = chosen_AIF.split()
                quant_model['AIF_number'] = [int(a) for a in quant_model['AIF_number']]

                if quant_model['AIF_number'] == [len(quant_model['AIFs'])]:
                    quant_model['AIF_number'] = np.arange(1, len(quant_model['AIFs']), 1, dtype=int)
                if all(q < (len(quant_model['AIFs'])) for q in quant_model['AIF_number']):
                    invalid_input = False
                else:
                    print("Number out of range")
            else:
                quant_model['AIF_number'] = []
                invalid_input = False

        invalid_input = True
        while invalid_input:
            if 2 in quant_model['AIF_number']:
                popAIF_dir = os.path.join(calc_dcm, "PopAIF")
                popAIFs_list = os.listdir(popAIF_dir)
                quant_model['PopAIFs'] = popAIFs_list
                quant_model['PopAIFs'].append("Use ALL")
                for i, popAIF in enumerate(popAIFs_list):
                    print(str(i+1)+":\t"+popAIF)
                chosen_popAIF = input("Which population AIF do you want to use for the pharmacokinetic analysis? \n>>> ")
                quant_model['PopAIF_number'] = chosen_popAIF.split()
                quant_model['PopAIF_number'] = [int(a) for a in quant_model['PopAIF_number']]

                if quant_model['PopAIF_number'] == [len(quant_model['PopAIFs'])]:
                    quant_model['PopAIF_number'] = np.arange(1, len(quant_model['PopAIFs']))
                if all(q < (len(quant_model['PopAIFs'])) for q in quant_model['PopAIF_number']):
                    invalid_input = False
                else:
                    print("Number out of range. Choose again.")
            else:
                invalid_input = False

    else:
        quant_model['AIF_number'] = []

    if 5 in quant_model['quantity_number']:
        chosen_AUC = input("Choose AUC value (seperate values with <space> and"
                           "<enter> will give you the typical values 60, 90 and 120):\n>>> ")
        chosen_AUC = chosen_AUC.split()
        if chosen_AUC == []:
            chosen_AUC = [60, 90, 120]
        quant_model['AUC_values'] = chosen_AUC

    if 7 in quant_model['quantity_number']:
        invalid_input = True
        while invalid_input:
            popAIF_dir = os.path.join(calc_dcm, "PopAIF")
            popAIF_temp_list = []
            for i, popAIF in enumerate(os.listdir(popAIF_dir)):
                print(str(i + 1) + ":\t" + popAIF)
                popAIF_temp_list.append(popAIF)
            popAIF_temp_list.append("All")
            quant_model['PopAIFs'] = popAIF_temp_list

            chosen_popAIF = input("Choose which population AIFs you want to calculate ICC for. \n>>> ")
            chosen_popAIF = chosen_popAIF.split()
            quant_model['PopAIF_number'] = [int(n) for n in chosen_popAIF]

            if quant_model['PopAIF_number'] == [len(quant_model['PopAIFs'])]:
                quant_model['PopAIF_number'] = np.arange(1, len(quant_model['PopAIFs']), 1, dtype=int)
            if all(p < (len(quant_model['PopAIFs'])) for p in quant_model['PopAIF_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")

    # Correlation calculation without outliers
    if 7 in quant_model['quantity_number'] or 8 in quant_model['quantity_number']:
        invalid_input = True
        while invalid_input:
            for i in range(len(quant_model['modules'])):
                print(str(i + 1) + ":\t" + quant_model['modules'][i])
            chosen_modules = input("Which module do you want to use for the CC calculations? \n>>> ")
            quant_model['module_number'] = chosen_modules.split()
            quant_model['module_number'] = [int(a) for a in quant_model['module_number']]

            if quant_model['module_number'] == [len(quant_model['modules'])]:
                quant_model['module_number'] = np.arange(1, len(quant_model['modules']), 1, dtype=int)
            if all(q < (len(quant_model['modules'])) for q in quant_model['module_number']):
                invalid_input = False
            else:
                print("Number out of range")

        invalid_input = True
        while invalid_input:
            for i in range(len(quant_model['tumour'])):
                print(str(i + 1) + ":\t" + quant_model['tumour'][i])
            chosen_modules = input("Which tumour types do you want to use for the CC calculations? \n>>> ")
            quant_model['tumour_number'] = chosen_modules.split()
            quant_model['tumour_number'] = [int(a) for a in quant_model['tumour_number']]

            if quant_model['tumour_number'] == [len(quant_model['tumour'])]:
                quant_model['tumour_number'] = np.arange(1, len(quant_model['tumour']), 1, dtype=int)
            if all(q < (len(quant_model['tumour'])) for q in quant_model['tumour_number']):
                invalid_input = False
            else:
                print("Number out of range")
    return quant_model


def choose_patients(data_dcm, quant_model, available_patients, calc=True, plot=False):
    """
    Function that lets the user choose which patients to do calculations for and which ones to include in plots.
    :param data_dcm: path to data to know which patients are available
    :param quant_model: dictionary with the quantities and models chosen by the user
    :param calc: choose patients for calculations
    :param plot: choose patients for plotting
    :return: a dictionary with the chosen patients for different actions
    """

    patient_numbers = {'PopAIF': [], 'Params': [], 'plots': [], 'Correlation': []}

    # Print all available patients
    for patient in available_patients:
        print(patient)
        patient_numbers['PopAIF'].append(patient)

    # Patients chosen for calculations
    if calc:

        if 3 in quant_model['quantity_number']:
            rm_patient_numbers = input("Which patients do you exclude for the calculations of population AIF "
                                   "(write the patient numbers (xx) seperate by space)? \n>>> ")
            rm_patient_numbers = rm_patient_numbers.split()
            for p in rm_patient_numbers:
                patient_numbers['PopAIF'].remove(p)
            print("Population AIF will be based on patient: {}".format(patient_numbers))


        if 1 in quant_model['quantity_number'] or 2 in quant_model['quantity_number'] \
            or 4 in quant_model['quantity_number'] or 5 in quant_model['quantity_number'] or 6 in \
            quant_model['quantity_number'] or 7 in quant_model['quantity_number'] or 8 in quant_model['quantity_number']\
                or 9 in quant_model['quantity_number']:

            invalid_input = True
            while invalid_input:
                r_or_a = input("Do you want to remove (<r>) or add (<a>) patients to do calculations for "
                               "or do it for all patients (<all>)? \n>>> ")

                if r_or_a == 'r' or r_or_a == 'R':
                    for name in sorted(os.listdir(data_dcm)):
                        if name[-4:] == "EMIN":
                            patient_numbers['Params'].append(name[:4])

                    invalid_patient_numbers = True
                    while invalid_patient_numbers:
                        chosen_patient_number = input("Which patients do you want to remove from the calculations (write the patient"
                                               " numbers (xxxx) separated by space or <A> to choose all patients)? \n>>> ")
                        chosen_patient_number = chosen_patient_number.split()
                        if all(patient in available_patients for patient in chosen_patient_number):
                            for p in chosen_patient_number:
                                patient_numbers['Params'].remove(p)
                            invalid_patient_numbers = False
                        else:
                            print("You chose a patient number(s) that does not exist. Choose again.")
                    invalid_input = False

                elif r_or_a == 'a' or r_or_a == 'A':
                    invalid_patient_numbers = True
                    while invalid_patient_numbers:
                        chosen_patient_number = input(
                            "Which patients do you choose for the calculations (write the patient numbers (xxxx) "
                            "separated by space) or choose all patients(<A>))? \n>>> ")
                        chosen_patient_number = chosen_patient_number.split()
                        if all(patient in available_patients  for patient in chosen_patient_number):
                            patient_numbers['Params'] = chosen_patient_number
                            invalid_patient_numbers = False
                        else:
                            print("You chose a patient number(s) that does not exist. Choose again.")
                    invalid_input = False

                elif r_or_a == 'all' or r_or_a == 'All' or r_or_a == 'ALL':
                    for name in sorted(os.listdir(data_dcm)):
                        if name[-4:] == "EMIN":
                            patient_numbers['Params'].append(name[:4])
                    print('Parameter calculations will be done for ALL patients')
                    invalid_input = False

                else:
                    print("You need to choose r (remove), a (add) or all.")

        """
        if 8 in quant_model['quantity_number'] or 9 in quant_model['quantity_number']:
            patient_number = input(
                "Which patients do you want to include in the correlation calculations (write the patient numbers (xx)"
                "seperated by <space>)? \n>>> ")
            patient_number = patient_number.split()
            patient_numbers["Correlation"] = patient_number
            print("The correlation calculations will be based on patients: {}".format(patient_number))
        """

    # Patients chosen for plotting
    if plot:
        invalid_input = True
        while invalid_input:
            r_or_a = input("Do you want to remove (<r>) or add (<a>) patients to do calculations for "
                                   "or do it for all patients (<all>)? \n>>> ")

            if r_or_a == 'r' or r_or_a == 'R':
                for name in sorted(os.listdir(data_dcm)):
                    if name[-4:] == "EMIN":
                        patient_numbers['plots'].append(name[:4])

                invalid_patient_numbers = True
                while invalid_patient_numbers:
                    chosen_patient_number = input(
                        "Which patients do you want to remove from the plots (write the patient"
                        " numbers (xxxx) separated by space or <A> to choose all patients)? \n>>> ")
                    chosen_patient_number = chosen_patient_number.split()
                    if all(patient in available_patients for patient in chosen_patient_number):
                        for p in chosen_patient_number:
                            patient_numbers['plots'].remove(p)
                        invalid_patient_numbers = False
                    else:
                        print("You chose a patient number(s) that does not exist. Choose again.")
                invalid_input = False

            elif r_or_a == 'a' or r_or_a == 'A':
                invalid_patient_numbers = True
                while invalid_patient_numbers:
                    chosen_patient_number = input(
                        "Which patients do you choose for the plots (write the patient numbers (xxxx) "
                        "separated by space) or choose all patients(<A>))? \n>>> ")
                    chosen_patient_number = chosen_patient_number.split()
                    if all(patient in available_patients for patient in chosen_patient_number):
                        patient_numbers['plots'] = chosen_patient_number
                        invalid_patient_numbers = False
                    else:
                        print("You chose a patient number(s) that does not exist. Choose again.")
                invalid_input = False

            elif r_or_a == 'all' or r_or_a == 'All' or r_or_a == 'ALL':
                for name in sorted(os.listdir(data_dcm)):
                    if name[-4:] == "EMIN":
                        patient_numbers['plots'].append(name[:4])
                print('ALL patients will be included in the plots ')
                invalid_input = False

            else:
                print("You need to choose r (remove), a (add) or all.")

    return patient_numbers

def choose_plots(calc_dcm):
    plot_model = {"plots": ["IndAIF validation", "PopAIF validation", "Parameter validation", "IndAIF vs PopAIF",
                            "PopAIF comparison", "Parameter comparison - IndAIF vs PopAIF", "Parameter comparison \w CCC",
                            "Parameter comparison \w Pearson and Spearman CC", "Heatmaps of CCs",  "Fitting comparison",
                            "Box plot of parameters TNM staging",
                            "Parameter comparison - IndAIF vs all PopAIF in one plot",
                            "Heapmaps of Pearson CCs without outliers",
                            "Parameter comparison \w Pearson CC \wo outliers",
                            "Parameter comparison \w CCC \wo outliers",
                            "Three heatmaps of Pearson CCs without outliers",
                            "Four heatmaps of Pearson CCs without outliers",
                            "Multiple box plots of parameters T stage", "Parameter histogram", "All"],
                  "models": ["TM", "ETM", "Brix", "All"],
                  "parameters": ['Kep_Brix', 'Kel_Brix', 'A_Brix', 'Kt_TM', 've_TM', 'Kep_TM',
                         'Kt_ETM', 've_ETM', 'vp_ETM', 'Kep_ETM', 'AUC60', 'AUC90', 'AUC120', 'TTHP', "All"],
                  "modules": ['VbyV', 'ROI', 'mean', 'median', "All"],
                  "tumour type": ["primary", "nodal", "nANDp", "All"],
                  "HPV type": ["T_positive", "T_negative", "T_both", "N_positive", "N_negative", "N_both", "All"],
                  "PopAIF_type": ['PopAIF', 'PopAIFrmBl', 'PopAIFpa', 'PopAIFpaRmBl', 'PopAIFwia', 'PopAIFwiaRmBl', 'All']}

    # Print the different plotting options in the console
    for i in range(len(plot_model['plots'])):
        print(str(i + 1) + ":\t" + plot_model['plots'][i])

    # User chooses what to plot
    invalid_input = True
    while invalid_input:
        chosen_plots = input("What plots do you want to create (seperated by <space>)? \n>>> ")
        plot_model['plot_number'] = chosen_plots.split()
        plot_model['plot_number'] = [int(a) for a in plot_model['plot_number']]
        if plot_model['plot_number'] == [len(plot_model['plots'])]:
            plot_model['plot_number'] = np.arange(1, len(plot_model['plots']), 1, dtype=int)
        if all(p < (len(plot_model['plots'])) for p in plot_model['plot_number']):
            invalid_input = False
        else:
            print("One or more numbers were out of range. Choose again.")

    if 4 in plot_model['plot_number'] or 5 in plot_model['plot_number'] or 6 in plot_model['plot_number'] \
            or 12 in plot_model['plot_number']:
        invalid_input = True
        while invalid_input:
            popAIF_dir = os.path.join(calc_dcm, "PopAIF")
            popAIF_temp_list = []
            for i, popAIF in enumerate(os.listdir(popAIF_dir)):
                print(str(i + 1) + ":\t" + popAIF)
                popAIF_temp_list.append(popAIF)
            popAIF_temp_list.append("All")
            plot_model['PopAIFs'] = popAIF_temp_list

            chosen_popAIF = input("Choose which population AIF you want to plot vs individual AIFs. \n>>> ")
            chosen_popAIF = chosen_popAIF.split()
            plot_model['PopAIF_number'] = [int(n) for n in chosen_popAIF]

            if plot_model['PopAIF_number'] == [len(plot_model['PopAIFs'])]:
                plot_model['PopAIF_number'] = np.arange(1, len(plot_model['PopAIFs']), 1, dtype=int)
            if all(p < (len(plot_model['PopAIFs'])) for p in plot_model['PopAIF_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")

    if 6 in plot_model['plot_number'] or 12 in plot_model['plot_number']:
        invalid_input = True
        while invalid_input:
            for i, popAIF in enumerate(plot_model['PopAIF_type']):
                print(str(i + 1) + ":\t" + popAIF)
            chosen_popAIF = input("Choose which population AIF you want to plot vs individual AIFs. \n>>> ")
            chosen_popAIF = chosen_popAIF.split()
            plot_model['PopAIF_type_number'] = [int(n) for n in chosen_popAIF]

            if plot_model['PopAIF_type_number'] == [len(plot_model['PopAIF_type'])]:
                plot_model['PopAIF_type_number'] = np.arange(1, len(plot_model['PopAIF_type']), 1, dtype=int)
            if all(p < (len(plot_model['PopAIF_type'])) for p in plot_model['PopAIF_type_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")


    if 6 in plot_model['plot_number'] or 7 in plot_model['plot_number'] or 8 in plot_model['plot_number'] \
            or 9 in plot_model['plot_number'] or 12 in plot_model['plot_number']:
        invalid_input = True
        while invalid_input:
            for i, model in enumerate(plot_model['parameters']):
                print(str(i+1) + ":\t" + model)
            chosen_models = input("Choose which parameters to do CCC calculations for. \n >>>")
            chosen_models = chosen_models.split()
            plot_model['param_number'] = [int(n) for n in chosen_models]

            if plot_model['param_number'] == [len(plot_model['parameters'])]:
                plot_model['param_number'] = np.arange(1, len(plot_model['parameters']), 1, dtype=int)
            if all(p < (len(plot_model['parameters'])) for p in plot_model['param_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")

        invalid_input = True
        while invalid_input:
            for i, model in enumerate(plot_model['modules']):
                print(str(i + 1) + ":\t" + model)
            chosen_models = input("Choose which modules to use for the CCC calculations. \n >>>")
            chosen_models = chosen_models.split()
            plot_model['module_number'] = [int(n) for n in chosen_models]

            if plot_model['module_number'] == [len(plot_model['modules'])]:
                plot_model['module_number'] = np.arange(1, len(plot_model['modules']), 1, dtype=int)
            if all(p < (len(plot_model['modules'])) for p in plot_model['module_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")

        invalid_input = True
        while invalid_input:

            for i, tumour_type in enumerate(plot_model['tumour type']):
                print(str(i + 1) + ":\t" + tumour_type)
            chosen_models = input("Which tumour types do you want to include in the CCC calculations "
                                  "(separate by <space>)? \n >>>")


            chosen_models = chosen_models.split()
            plot_model['tumour_number'] = [int(n) for n in chosen_models]

            if plot_model['tumour_number'] == [len(plot_model['tumour type'])]:
                plot_model['tumour_number'] = np.arange(1, len(plot_model['tumour type']), 1, dtype=int)
            if all(p < (len(plot_model['tumour type'])) for p in plot_model['tumour_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")

    if 10 in plot_model['plot_number']:
        invalid_input = True
        while invalid_input:
            for i, model in enumerate(plot_model['models']):
                print(str(i + 1) + ":\t" + model)
            chosen_models = input("Choose which model to do fitting with. \n>>> ")
            chosen_models = chosen_models.split()
            plot_model['model_number'] = [int(n) for n in chosen_models]

            if plot_model['model_number'] == [len(plot_model['models'])]:
                plot_model['model_number'] = np.arange(1, len(plot_model['models']), 1, dtype=int)
            if all(p < (len(plot_model['models'])) for p in plot_model['model_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")

    if 11 in plot_model['plot_number'] or 18 in plot_model['plot_number']:
        invalid_input = True
        while invalid_input:
            for i, model in enumerate(plot_model['parameters']):
                print(str(i + 1) + ":\t" + model)
            chosen_models = input("Choose which parameters to do TNM correlation for. \n >>>")
            chosen_models = chosen_models.split()
            plot_model['param_number'] = [int(n) for n in chosen_models]

            if plot_model['param_number'] == [len(plot_model['parameters'])]:
                plot_model['param_number'] = np.arange(1, len(plot_model['parameters']), 1, dtype=int)
            if all(p < (len(plot_model['parameters'])) for p in plot_model['param_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")

        invalid_input = True
        while invalid_input:
            for i, model in enumerate(plot_model['HPV type']):
                print(str(i + 1) + ":\t" + model)
            chosen_models = input("Choose which HPV type to create box plot for. \n>>> ")
            chosen_models = chosen_models.split()
            plot_model['HPV_type_number'] = [int(n) for n in chosen_models]

            if plot_model['HPV_type_number'] == [len(plot_model['HPV type'])]:
                plot_model['HPV_type_number'] = np.arange(1, len(plot_model['HPV type']), 1, dtype=int)
            if all(p < (len(plot_model['HPV type'])) for p in plot_model['HPV_type_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")

    if 13 in plot_model['plot_number'] or 14 in plot_model['plot_number'] or 15 in plot_model['plot_number']\
            or 16 in plot_model['plot_number'] or 17 in plot_model['plot_number']:
        """
        invalid_input = True
        while invalid_input:
            for i, model in enumerate(plot_model['modules']):
                print(str(i + 1) + ":\t" + model)
            chosen_models = input("Choose which modules to use for the CCC calculations. \n >>>")
            chosen_models = chosen_models.split()
            plot_model['module_number'] = [int(n) for n in chosen_models]

            if plot_model['module_number'] == [len(plot_model['modules'])]:
                plot_model['module_number'] = np.arange(1, len(plot_model['modules']), 1, dtype=int)
            if all(p < (len(plot_model['modules'])) for p in plot_model['module_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")

        invalid_input = True
        while invalid_input:

            for i, tumour_type in enumerate(plot_model['tumour type']):
                print(str(i + 1) + ":\t" + tumour_type)
            chosen_models = input("Which tumour types do you want to include in the CCC calculations "
                                  "(separate by <space>)? \n >>>")

            chosen_models = chosen_models.split()
            plot_model['tumour_number'] = [int(n) for n in chosen_models]

            if plot_model['tumour_number'] == [len(plot_model['tumour type'])]:
                plot_model['tumour_number'] = np.arange(1, len(plot_model['tumour type']), 1, dtype=int)
            if all(p < (len(plot_model['tumour type'])) for p in plot_model['tumour_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")
        """
        invalid_input = True
        while invalid_input:
            CC_dir = os.path.join(calc_dcm, "Correlation_wo_outliers")
            CC_list = os.listdir(CC_dir)
            plot_model['CC_matrices'] = CC_list
            plot_model['CC_matrices'].append("Use ALL")
            for i, CC_matrix in enumerate(CC_list):
                print(str(i + 1) + ":\t" + CC_matrix)
            chosen_CC_matrix = input("Which correlation matrix do you want to plot? \n>>> ")
            plot_model['CC_number'] = chosen_CC_matrix.split()
            plot_model['CC_number'] = [int(a) for a in plot_model['CC_number']]

            if plot_model['CC_number'] == [len(plot_model['CC_matrices'])]:
                plot_model['CC_number'] = np.arange(1, len(plot_model['CC_matrices']))
            if all(q < (len(plot_model['CC_matrices'])) for q in plot_model['CC_number']):
                invalid_input = False
            else:
                print("Number out of range. Choose again.")

    if 14 in plot_model['plot_number'] or 15 in plot_model['plot_number']:
         invalid_input = True
         while invalid_input:
             for i, model in enumerate(plot_model['parameters']):
                 print(str(i + 1) + ":\t" + model)
             chosen_models = input("Choose which parameters to do CCC calculations for. \n >>>")
             chosen_models = chosen_models.split()
             plot_model['param_number'] = [int(n) for n in chosen_models]

             if plot_model['param_number'] == [len(plot_model['parameters'])]:
                 plot_model['param_number'] = np.arange(1, len(plot_model['parameters']), 1, dtype=int)
             if all(p < (len(plot_model['parameters'])) for p in plot_model['param_number']):
                 invalid_input = False
             else:
                 print("One or more numbers were out of range. Choose again.")
    if 19 in plot_model['plot_number']:
        invalid_input = True
        while invalid_input:
            for i, model in enumerate(plot_model['parameters']):
                print(str(i + 1) + ":\t" + model)
            chosen_models = input("Choose which parameters to do histogram for. \n >>>")
            chosen_models = chosen_models.split()
            plot_model['param_number'] = [int(n) for n in chosen_models]

            if plot_model['param_number'] == [len(plot_model['parameters'])]:
                plot_model['param_number'] = np.arange(1, len(plot_model['parameters']), 1, dtype=int)
            if all(p < (len(plot_model['parameters'])) for p in plot_model['param_number']):
                invalid_input = False
            else:
                print("One or more numbers were out of range. Choose again.")


    return plot_model





