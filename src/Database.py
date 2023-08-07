################################################################################
######################## I M P O R T  P A C K A G E S ##########################
################################################################################

import os
import numpy as np
from src import Modelling
from scipy.optimize import curve_fit
import pydicom as pd
from scipy.stats import spearmanr, pearsonr, linregress
import pandas as pd
import pyirr

################################################################################
################# P A T I E N T  D A T A B A S E  C L A S S ####################
################################################################################

class PatientDatabase():
    """
    A class that will contain the patient data.
    """
    def __init__(self, data_path, calc_path, time, sek_or_min, T1, R_Gd, TR, FA, injection_time, injection_speed, TA, dt):
        # Constants
        self.T1 = T1
        self.R_Gd = R_Gd
        self.TR = TR
        self.FA = FA
        self.injection_time = injection_time
        self.injection_speed = injection_speed
        self.TA = TA
        self.dt = dt

        # Time
        self.time = time
        self.sek_or_min = sek_or_min

        # Paths
        self.data_path = data_path
        self.calc_path = calc_path

        # Patients
        self.available_patients = self.get_available_patients()
        self.chosen_patients = []
        self.chosen_popAIF_patients = []




    def get_available_patients(self):
        """
        A function that retrieves the id-number of the available patients
        :return: a list of patient numbers
        """
        patients = []
        patient_folders = os.listdir(self.data_path)
        for folder in patient_folders:
            if folder[5:9] == "EMIN":
                patient = folder[:4]
                patients.append(patient)

        patient = sorted(patients)
        return patient

    def add_patients(self, patients, AIF_slices, AIF_times):
        for patient in patients:
            AIF_slice = AIF_slices[patient]
            AIF_time = AIF_times[patient]
            self.chosen_patients.append(Patient(self.data_path, self.calc_path, patient, AIF_slice, AIF_time,
                                                self.injection_speed, self.injection_time, self.T1, self.R_Gd, self.TR,
                                                self.FA, self.time, self.sek_or_min))

    def add_popAIF_patients(self, patients, AIF_slices, AIF_times):
        for patient in patients:
            AIF_slice = AIF_slices[patient]
            AIF_time = AIF_times[patient]
            self.chosen_popAIF_patients.append(Patient(self.data_path, self.calc_path, patient, AIF_slice, AIF_time,
                                                       self.injection_speed, self.injection_time, self.T1, self.R_Gd,
                                                       self.TR, self.FA, self.time, self.sek_or_min))
        self.popAIF_patient_numbers = patients

    def does_popAIF_exists(self):
        patients = self.popAIF_patient_numbers
        popAIF_path = os.path.join(self.calc_path, "PopAIF/popAIF_p{}".format(patients))
        return os.path.exists(popAIF_path)


    def calculate_popAIF(self):
        # Create a folder where population AIFs will be stored
        popAIF_dir = os.path.join(self.calc_path, "PopAIF")
        if os.path.exists(popAIF_dir) == False:
            os.makedirs(popAIF_dir)

        # Time arrays
        time = np.arange(0, 60 * 3.735, 3.735)
        #time_rm_bl = time[self.injection_time:] - time[self.injection_time] # Baseline removed (the time before injection)
        time_rm_bl = time[self.injection_time:]

        # Retrieve the mean of the individual AIFs that the population AIF will be based on
        indAIF_list = np.zeros((len(self.popAIF_patient_numbers), 60))
        T1_list = np.zeros(len(self.chosen_popAIF_patients))
        for p_indx, Patient in enumerate(self.chosen_popAIF_patients):
            try:
                indAIF_list[p_indx, :] = Patient.IndAIF
                T1_list[p_indx] = Patient.IndAIF_T1


            except:
                if os.path.exists(Patient.IndAIF_path) == False:
                    Patient.calculate_IndAIF(self.T1, self.injection_time, self.R_Gd, self.TR, self.FA)

                else:
                    IndAIF_data = np.load(Patient.IndAIF_path)
                    IndAIF_signal = IndAIF_data["AIF_signal"] # IndAIF before it has been converted to concentration
                    indAIF_list[p_indx, :] = IndAIF_signal
                    T1_list[p_indx] = IndAIF_data['T1']

        # Average T1 to be used for population AIF concentration calculation
        T1_mean = np.mean(T1_list)

        indAIF_list = np.nan_to_num(indAIF_list)

        # Calculate six different population AIFs: no alignment + baseline, no alignment - baseline,
        #                                          peak alignment + baseline, peak alignment - baseline
        #                                          wash in alignment + baseline, wash in alignment - baselin

        popAIFs, popAIF_params, popAIF_params_std, new_peak_indx, new_wi_indx = Modelling.calculate_popAIFs(indAIF_list,
                                                                                time, time_rm_bl, self.injection_time,
                                                                                self.R_Gd, self.TR, self.FA, T1_mean)

        # Lagre population AIFs i numpy file
        self.popAIF_loc = os.path.join(popAIF_dir, "popAIF_p{}_.npz".format(self.popAIF_patient_numbers))
        np.savez(self.popAIF_loc, patients=self.popAIF_patient_numbers, T1=T1_mean,
                 AIF=popAIFs[0], popt=popAIF_params[0], std=popAIF_params_std[0],
                 AIF_rm_bl=popAIFs[1], popt_rm_bl=popAIF_params[1], std_rm_bl=popAIF_params_std[1],
                 AIF_aligned_peak=popAIFs[2], popt_aligned_peaks=popAIF_params[2],
                 std_aligned_peak=popAIF_params_std[2], peak_indx=new_peak_indx,
                 AIF_aligned_peak_rm_bl=popAIFs[3], popt_aligned_peaks_rm_bl=popAIF_params[3],
                 std_aligned_peak_rm_bl=popAIF_params_std[3],
                 AIF_aligned_wi=popAIFs[4], popt_aligned_wi=popAIF_params[4], std_aligned_wi=popAIF_params_std[4],
                 wi_indx=new_wi_indx,
                 AIF_aligned_wi_rm_bl=popAIFs[5], popt_aligned_wi_rm_bl=popAIF_params[5],
                 std_aligned_wi_rm_bl=popAIF_params_std[5])

    def retrieve_parameters_from_all_patients(self, parameter_numbers, parameters):
        # Gather data to do correlation calculations with
        parameter_data_list = [[] for i in range(12)]
        module = ['VbyV', 'ROI', 'mean', 'median', 'VbyV', 'ROI', 'mean', 'median', 'VbyV', 'ROI', 'mean', 'median']
        tumour_type_list = ['primary', 'primary', 'primary', 'primary', 'nodal', 'nodal', 'nodal', 'nodal',
                            'nANDp', 'nANDp', 'nANDp', 'nANDp']

        parameter_numbers = parameter_numbers-np.ones(len(parameter_numbers), dtype=int)
        patient_GTVn_list = []
        patient_GTVp_list = []
        for Patient in self.chosen_patients:
            Patient.add_tumours()

            # Check if the parameters have been calculated and if not calculate the parameters
            all_parameters_exists, new_model_calculations = Patient.all_parameters_exist()
            if all_parameters_exists == False:
                for model in new_model_calculations:
                    if model == 'AUC':
                        AUC_time_values = [60, 90, 120]
                        Patient.calculate_AUC(AUC_time_values, self.TA, self.dt)

                    else:
                        quant_model = {'AIF_number': [1]}
                        Patient.calculate_pharmacokinetic_parameters(model, quant_model)

            correlation_dir = os.path.join(Patient.patient_calc_path, "Correlation")
            if os.path.exists(correlation_dir) == False:
                os.makedirs(correlation_dir)

            parameter_data_VbyV_primary_patient, parameter_data_ROI_primary_patient, \
            parameter_data_mean_primary_patient, parameter_data_median_primary_patient \
                = Patient.retrieve_parameters(parameters, 'p')

            if parameter_data_ROI_primary_patient != []:
                parameter_data_ROI_primary_patient = parameter_data_ROI_primary_patient[:, parameter_numbers]
                patient_GTVp_list.append(Patient.patient_name)

            parameter_data_VbyV_nodal_patient, parameter_data_ROI_nodal_patient, \
            parameter_data_mean_nodal_patient, parameter_data_median_nodal_patient \
                = Patient.retrieve_parameters(parameters, 'n')

            if parameter_data_ROI_nodal_patient != []:
                parameter_data_ROI_nodal_patient = parameter_data_ROI_nodal_patient[:, parameter_numbers]
                for t in Patient.chosen_tumour_names:
                    if t[3] == 'n':
                        patient_GTVn_list.append(Patient.patient_name)

            if parameter_data_VbyV_nodal_patient != [] and parameter_data_VbyV_primary_patient != []:
                parameter_data_VbyV_nANDp_patient = np.concatenate((parameter_data_VbyV_primary_patient,
                                                                    parameter_data_VbyV_nodal_patient), axis=0)
                parameter_data_ROI_nANDp_patient = np.concatenate((parameter_data_ROI_primary_patient,
                                                                   parameter_data_ROI_nodal_patient), axis=0)
                parameter_data_mean_nANDp_patient = np.concatenate((parameter_data_mean_primary_patient,
                                                                    parameter_data_mean_nodal_patient), axis=0)
                parameter_data_median_nANDp_patient = np.concatenate((parameter_data_median_primary_patient,
                                                                      parameter_data_median_nodal_patient), axis=0)

            elif parameter_data_VbyV_primary_patient != []:
                parameter_data_VbyV_nANDp_patient = parameter_data_VbyV_primary_patient
                parameter_data_ROI_nANDp_patient = parameter_data_ROI_primary_patient
                parameter_data_mean_nANDp_patient = parameter_data_mean_primary_patient
                parameter_data_median_nANDp_patient = parameter_data_median_primary_patient

            elif parameter_data_VbyV_nodal_patient != []:
                parameter_data_VbyV_nANDp_patient = parameter_data_VbyV_nodal_patient
                parameter_data_ROI_nANDp_patient = parameter_data_ROI_nodal_patient
                parameter_data_mean_nANDp_patient = parameter_data_mean_nodal_patient
                parameter_data_median_nANDp_patient = parameter_data_median_nodal_patient

            else:
                parameter_data_VbyV_nANDp_patient, parameter_data_ROI_nANDp_patient, parameter_data_mean_nANDp_patient, parameter_data_median_nANDp_patient = [], [], [], []

            parameter_data_patient_list = [parameter_data_VbyV_primary_patient, parameter_data_ROI_primary_patient,
                                           parameter_data_mean_primary_patient, parameter_data_median_primary_patient,
                                           parameter_data_VbyV_nodal_patient, parameter_data_ROI_nodal_patient,
                                           parameter_data_mean_nodal_patient, parameter_data_median_nodal_patient,
                                           parameter_data_VbyV_nANDp_patient, parameter_data_ROI_nANDp_patient,
                                           parameter_data_mean_nANDp_patient, parameter_data_median_nANDp_patient]

            # Adding the patient data to the matrix storing all patient data
            for i, parameter_data in enumerate(parameter_data_list):

                if len(parameter_data_patient_list[i]) > 0:
                    if parameter_data == []:
                        parameter_data_list[i] = parameter_data_patient_list[i]

                    elif parameter_data != []:
                        parameter_data_list[i] = np.concatenate(
                            (parameter_data_list[i], parameter_data_patient_list[i]), axis=0)

        return parameter_data_list, patient_GTVn_list, patient_GTVp_list

    def retrieve_IndAIF_and_PopAIF_parameters_from_patients(self, parameter, model, popAIF_type, popAIF_patients, tumour_type, module):
        values = np.array([])
        for i, Patient in enumerate(self.chosen_patients):
            Patient.add_tumours()
            if parameter == 'Kep':
                Ktrans_values = Patient.retrieve_parameter('Kt', model, popAIF_type, popAIF_patients,
                                                              tumour_type, module)
                ve_values = Patient.retrieve_parameter('ve', model, popAIF_type, popAIF_patients,
                                                              tumour_type, module)
                parameter_values = np.array(Ktrans_values)/np.array(ve_values)

            else:
                parameter_values = Patient.retrieve_parameter(parameter, model, popAIF_type, popAIF_patients, tumour_type, module)
                parameter_values = np.array(parameter_values)

            if len(parameter_values) != 0:
                values = np.append(values, parameter_values)

        return values


    def calculate_correlations_wo_outliers(self, parameter_names, module, tumour_type):
        correlation_matrix = np.zeros((len(parameter_names), len(parameter_names)))
        pvalue_matrix = np.zeros((len(parameter_names), len(parameter_names)))
        slope_matrix = np.zeros((len(parameter_names), len(parameter_names)))
        intercept_matrix = np.zeros((len(parameter_names), len(parameter_names)))
        slope_std_matrix = np.zeros((len(parameter_names), len(parameter_names)))
        intercept_std_matrix = np.zeros((len(parameter_names), len(parameter_names)))
        removed_outliers = np.zeros((len(parameter_names), len(parameter_names)))
        for i, parameter1_metadata in enumerate(parameter_names):
            parameter1_metadata = parameter1_metadata.split("_")
            parameter1_name = parameter1_metadata[0]
            try:
                model1 = parameter1_metadata[1]
            except:
                if parameter1_metadata[0][:3] == 'AUC':
                    model1 = 'AUC'
                else:
                    model1 = parameter1_metadata[0]
            for j, parameter2_metadata in enumerate(parameter_names):
                parameter2_metadata = parameter2_metadata.split("_")
                parameter2_name = parameter2_metadata[0]
                try:
                    model2 = parameter2_metadata[1]
                except:
                    if parameter2_metadata[0][:3] == 'AUC':
                        model2 = 'AUC'
                    else:
                        model2 = parameter2_metadata[0]

                parameter1_list = np.array([])
                parameter2_list = np.array([])
                for Patient in self.chosen_patients:
                    Patient.add_tumours()
                    for tumour in Patient.chosen_tumour_names:
                        if tumour_type == 'nANDp' or tumour[3] == tumour_type[0]:
                            parameter1 = Patient.retrieve_single_ROI_parameter(parameter1_name, model1, tumour, AIF="IndAIF")
                            parameter2 = Patient.retrieve_single_ROI_parameter(parameter2_name, model2, tumour, AIF="IndAIF")

                            parameter1_list = np.append(parameter1_list, parameter1)
                            parameter2_list = np.append(parameter2_list, parameter2)

                parameter1_list, parameter2_list, outliers = Modelling.remove_outliers(parameter1_list, parameter2_list)
                removed_outliers[i, j] = outliers

                res = linregress(parameter1_list, parameter2_list)
                correlation_matrix[i, j] = res.rvalue
                pvalue_matrix[i, j] = res.pvalue
                slope_matrix[i, j] = res.slope
                intercept_matrix[i, j] = res.intercept
                slope_std_matrix[i, j] = res.stderr
                intercept_std_matrix[i, j] = res.intercept_stderr
        return correlation_matrix, pvalue_matrix, slope_matrix, slope_std_matrix, intercept_matrix, intercept_std_matrix, removed_outliers
    def calculate_correlations(self, parameters):
        # Gather data to do correlation calculations with
        parameter_data_list = [[] for i in range(12)]
        module = ['VbyV', 'ROI', 'mean', 'median', 'VbyV', 'ROI', 'mean', 'median', 'VbyV', 'ROI', 'mean', 'median']
        tumour_type_list = ['primary', 'primary', 'primary', 'primary', 'nodal', 'nodal', 'nodal', 'nodal',
                            'nANDp', 'nANDp', 'nANDp', 'nANDp']

        for Patient in self.chosen_patients:
            Patient.add_tumours()

            # Check if the parameters have been calculated and if not calculate the parameters
            all_parameters_exists, new_model_calculations = Patient.all_parameters_exist()
            if all_parameters_exists == False:
                for model in new_model_calculations:
                    print("New parameters from model {} for: {}".format(model, Patient.patient_name))
                    if model == 'AUC':
                        AUC_time_values = [60, 90, 120]
                        Patient.calculate_AUC(AUC_time_values, self.TA, self.dt)

                    else:
                        quant_model = {'AIF_number': [1]}
                        Patient.calculate_pharmacokinetic_parameters(model, quant_model)

            correlation_dir = os.path.join(Patient.patient_calc_path, "Correlation")
            if os.path.exists(correlation_dir) == False:
                os.makedirs(correlation_dir)

            parameter_data_VbyV_primary_patient, parameter_data_ROI_primary_patient, \
            parameter_data_mean_primary_patient, parameter_data_median_primary_patient \
            = Patient.retrieve_parameters(parameters, 'p')

            parameter_data_VbyV_nodal_patient, parameter_data_ROI_nodal_patient, \
            parameter_data_mean_nodal_patient, parameter_data_median_nodal_patient  \
            = Patient.retrieve_parameters(parameters, 'n')

            if parameter_data_VbyV_nodal_patient != [] and parameter_data_VbyV_primary_patient != []:
                parameter_data_VbyV_nANDp_patient = np.concatenate((parameter_data_VbyV_primary_patient,
                                                                    parameter_data_VbyV_nodal_patient), axis=0)
                parameter_data_ROI_nANDp_patient = np.concatenate((parameter_data_ROI_primary_patient,
                                                                    parameter_data_ROI_nodal_patient), axis=0)
                parameter_data_mean_nANDp_patient = np.concatenate((parameter_data_mean_primary_patient,
                                                                    parameter_data_mean_nodal_patient), axis=0)
                parameter_data_median_nANDp_patient = np.concatenate((parameter_data_median_primary_patient,
                                                                    parameter_data_median_nodal_patient), axis=0)

            elif parameter_data_VbyV_primary_patient != []:
                parameter_data_VbyV_nANDp_patient = parameter_data_VbyV_primary_patient
                parameter_data_ROI_nANDp_patient = parameter_data_ROI_primary_patient
                parameter_data_mean_nANDp_patient = parameter_data_mean_primary_patient
                parameter_data_median_nANDp_patient = parameter_data_median_primary_patient

            elif parameter_data_VbyV_nodal_patient != []:
                parameter_data_VbyV_nANDp_patient = parameter_data_VbyV_nodal_patient
                parameter_data_ROI_nANDp_patient = parameter_data_ROI_nodal_patient
                parameter_data_mean_nANDp_patient = parameter_data_mean_nodal_patient
                parameter_data_median_nANDp_patient = parameter_data_median_nodal_patient

            else:
                parameter_data_VbyV_nANDp_patient, parameter_data_ROI_nANDp_patient, parameter_data_mean_nANDp_patient, parameter_data_median_nANDp_patient = [], [], [], []

            parameter_data_patient_list = [parameter_data_VbyV_primary_patient, parameter_data_ROI_primary_patient,
                                           parameter_data_mean_primary_patient, parameter_data_median_primary_patient,
                                           parameter_data_VbyV_nodal_patient, parameter_data_ROI_nodal_patient,
                                           parameter_data_mean_nodal_patient, parameter_data_median_nodal_patient,
                                           parameter_data_VbyV_nANDp_patient, parameter_data_ROI_nANDp_patient,
                                           parameter_data_mean_nANDp_patient, parameter_data_median_nANDp_patient]


            # Adding the patient data to the matrix storing all patient data
            for i, parameter_data in enumerate(parameter_data_list):

                if len(parameter_data_patient_list[i]) > 0:
                    if parameter_data == []:
                        parameter_data_list[i] = parameter_data_patient_list[i]

                    elif parameter_data != []:
                        parameter_data_list[i] = np.concatenate((parameter_data_list[i], parameter_data_patient_list[i]), axis=0)

                    # Calculate pearson and spearman correlations for the patient
                    if i == 0 or i == 4 or i == 8:
                        spearman_corr_matrix_patient, spearman_pval_matrix_patient = spearmanr(parameter_data_patient_list[i], axis=0)
                        pearsonr_corr_matrix_patient, pearsonr_pval_matrix_patient = Modelling.pearsonr(parameter_data_patient_list[i].T)

                        correlation_path = os.path.join(correlation_dir, "Correlation_{}_{}_tumour_."
                                                                         "npz".format(module[i], tumour_type_list[i]))
                        np.savez(correlation_path,
                                 Spearman=spearman_corr_matrix_patient, Spearman_pval=spearman_pval_matrix_patient,
                                 Pearson=pearsonr_corr_matrix_patient, Pearson_pval=pearsonr_pval_matrix_patient,
                                 parameters=parameter_data_patient_list, parameter_names=parameters)


        correlation_dir = os.path.join(self.calc_path, "Correlation")
        if os.path.exists(correlation_dir) == False:
            os.makedirs(correlation_dir)

        for i, parameter_data in enumerate(parameter_data_list):
            if len(parameter_data.T) > 1:
                spearman_corr_matrix, spearman_pval_matrix = spearmanr(parameter_data, axis=0)
                pearsonr_corr_matrix, pearsonr_pval_matrix = Modelling.pearsonr(parameter_data.T)

                correlation_path = os.path.join(correlation_dir, "Correlation_{}_{}_p{}_."
                                                            "npz".format(module[i], tumour_type_list[i],
                                                            [int(Patient.patient_name[:4])-1000 for Patient in self.chosen_patients]))
                np.savez(correlation_path, Spearman=spearman_corr_matrix, Spearman_pval=spearman_pval_matrix,
                         Pearson=pearsonr_corr_matrix, Pearson_pval=pearsonr_pval_matrix,
                         parameters=parameter_data, parameter_names=parameters, patients=self.chosen_patients)

    def ICC_calculation(self, parameter_metadata, tumour_type, module, popAIFs, popAIF_patients,
                        ICC_model, ICC_type, ICC_definition):
        parameter_metadata_list = parameter_metadata.split("_")
        parameter_name = parameter_metadata_list[0]
        try:
            model = parameter_metadata_list[1]
        except:
            model = "AUC"

        parameter_dict = {}
        for popAIF in popAIFs:
            parameter_list = np.array([])
            for Patient in self.chosen_patients:
                Patient.add_tumours()
                for tumour in Patient.chosen_tumour_names:
                    if tumour_type == "nANDp" or tumour[3] == tumour_type[0]:
                        parameter = Patient.retrieve_single_ROI_parameter(parameter_name, model, tumour,
                                                                          popAIF_patients=popAIF_patients, AIF=popAIF)
                        parameter_list = np.append(parameter_list, parameter)
            parameter_dict[popAIF] = parameter_list

        if parameter_name == 've':
            parameter_dict, outliers = Modelling.remove_outliers_above_one_for_multiple_parameters(parameter_dict)


        df = pd.DataFrame(data=parameter_dict)
        icc = pyirr.intraclass_correlation(df, ICC_model, ICC_definition, ICC_type)
        print("ICC for {} {} parameter, {} tumours".format(module, parameter_name, tumour_type))
        print(icc)
        return icc.value








################################################################################
######################## P A T I E N T   C L A S S #############################
################################################################################

class Patient():

    def __init__(self, data_path, calc_path, patient_number, AIF_slice, AIF_time, injection_speed, injection_time, T1, R_Gd, TR, FA,
                 time, sek_or_min):
        # Constants
        self.injection_speed = injection_speed
        self.injection_time = injection_time
        self.R_Gd = R_Gd
        self.TR = TR
        self.FA = FA
        self.time = time
        self.time_rm_bl = time[injection_time:] - time[injection_time]
        self.T1 = T1
        self.sek_or_min = sek_or_min

        self.patient_name = "{}_EMIN_{}_EMIN".format(patient_number, patient_number)
        self.patient_number = patient_number

        self.calc_path = calc_path
        self.patient_data_path = os.path.join(data_path, self.patient_name)
        self.patient_database_path = os.path.join(calc_path, "PatientDatabase")
        self.patient_calc_path = os.path.join(self.patient_database_path, self.patient_name)

        self.AIF_slice = AIF_slice
        self.AIF_time = AIF_time
        self.Old_IndAIF_path = os.path.join(self.patient_calc_path, "AIF/IndAIF_s{}_.npz".format(self.AIF_slice))
        self.IndAIF_path = os.path.join(self.patient_calc_path, "AIF/IndAIF_t{}_.npz".format(self.AIF_time))

    def retrieve_single_ROI_parameter(self, parameter, model, tumour, **kwargs):
        parameter_dict = {'Kt': 0, 've': 1, 'vp': 2, 'Kep': 0, 'Kel': 1, 'A': 2, 'AUC60': 0, 'AUC90': 1, 'AUC120': 2}

        if model[:3] == 'AUC':
            ROI_file = "AUC/ROI_AUC_tumour_{}_.npz".format(tumour)
        elif model[:4] == 'TTHP':
            ROI_file = "TTHP/ROI_TTHP_tumour_{}_.npz".format(tumour)

        elif model == 'Brix':
            ROI_file = "Parameters/{}/MeanROI_param_model_Brix_tumour_{}_rm_bl_.npz".format(model, tumour)
        else:
            if kwargs['AIF'] == 'IndAIF':
                ROI_file = "Parameters/{}/MeanROI_param_IndAIF_t{}_model_{}_tumour_{}_.npz".format(model, self.AIF_time, model, tumour)
            else:
                patients = kwargs['popAIF_patients']
                popAIF = kwargs['AIF']
                ROI_file = "Parameters/{}/MeanROI_param_{}_p{}_model_{}_tumour_{}_.npz".format(model, popAIF, patients, model, tumour)

        ROI_path = os.path.join(self.patient_calc_path, ROI_file)
        data = np.load(ROI_path)
        if parameter == 'Kep' and model != 'Brix':
            param = data['ROI_param'][parameter_dict['Kt']]/data['ROI_param'][parameter_dict['ve']]
        elif model == 'AUC':
            param = data['AUC'][parameter_dict[parameter]]
        elif model == 'TTHP':
            param = data['TTHP']
        else:
            param = data['ROI_param'][parameter_dict[parameter]]

        return param



    def retrieve_parameter(self, parameter, model, popAIF_type, popAIF_patients, tumour_type, module):
        parameter_dict = {'Kt': 0, 've': 1, 'vp': 2}
        tumour_dict = {'nodal': 'n', 'primary': 'p'}
        values = []
        for i, (tumour_name, tumour_mask) in enumerate(zip(self.chosen_tumour_names, self.chosen_tumour_segmentations)):
            if tumour_type == 'nANDp' or tumour_name[3] == tumour_dict[tumour_type]:
                if module == 'ROI':
                    if popAIF_type[:6] == 'IndAIF':
                        if len(popAIF_type)==6:
                            parameter_file = "Parameters/{}/MeanROI_param_IndAIF_t{}_model_{}_tumour_{}_.npz".format(model,
                                                                                            self.AIF_time, model, tumour_name)
                        else:
                            parameter_file = "Parameters/{}/MeanROI_param_IndAIF_t{}_model_{}_tumour_{}_rm_bl_.npz".format(
                                model, self.AIF_time, model, tumour_name)

                    else:
                        parameter_file = "Parameters/{}/MeanROI_param_{}_{}_model_{}_tumour_{}_.npz".format(model,
                                                                        popAIF_type, popAIF_patients, model, tumour_name)
                    parameter_path = os.path.join(self.patient_calc_path, parameter_file)
                    parameter_data = np.load(parameter_path)
                    parameter_ROI = parameter_data['ROI_param'][parameter_dict[parameter]]
                    values.append(parameter_ROI[0])

                elif module == 'VbyV':
                    # Finding the indices for where in the parameter map the tumour is
                    tumour_mask_flat = tumour_mask.flatten()
                    tumour_idxs = np.where(tumour_mask_flat == True)

                    # Retrieving the parameter map
                    if popAIF_type == 'IndAIF':
                        parameter_file = "Parameters/{}/{}_IndAIF_t{}_model_{}_tumour_{}_.nii.gz".format(model, parameter,
                                                                           self.AIF_time, model, tumour_name)
                    else:
                        parameter_file = "Parameters/{}/{}_{}_{}_model_{}_tumour_{}_.nii.gz".format(model, parameter,
                                                                 popAIF_type, popAIF_patients, model, tumour_name)

                    parameter_path = os.path.join(self.patient_calc_path, parameter_file)
                    parameter_map = Modelling.get_image_numpy_data(parameter_path)
                    parameter_map_flat = parameter_map.flatten()
                    parameter_segmented = parameter_map_flat[tumour_idxs]
                    if values == []:
                        values = np.array(values.append(parameter_segmented))

                    else:
                        values = np.concatenate(values, [parameter_segmented])

        return values





    def set_AIF_slice(self, AIF_slice):
        self.AIF_slice = AIF_slice

    def does_Ct_exist(self):
        Ct_path = os.path.join(self.patient_calc_path, "Concentrations/Ct_.npz".format(self.AIF_slice))
        return os.path.exists(Ct_path)

    def load_Ct(self):
        Ct_path = os.path.join(self.patient_calc_path, "Concentrations/Ct_.npz".format(self.AIF_slice))
        Ct_data = np.load(Ct_path)
        self.Ct = Ct_data['conc']
        self.S = Ct_data['signal']
        self.affine = Ct_data['affine']

    def calculate_Ct(self, injection_time, R_Gd, TR, FA):
        # Obtain signal and calculate concentration in tumour
        signal, conc, affine = Modelling.tumour_conc_calc(self.patient_data_path, injection_time, R_Gd, TR, FA)

        # Save data
        conc_folder_path = os.path.join(self.patient_calc_path, "Concentrations")
        conc_path = os.path.join(conc_folder_path, "Ct_.npz")
        if os.path.exists(conc_folder_path) == False:
            os.makedirs(conc_folder_path)

        self.Ct = conc
        self.S = signal
        self.affine = affine

        np.savez(conc_path, signal=signal, conc=conc, affine=affine)
        print("Ct for patient {} has been saved to {}".format(self.patient_number, conc_path))

    def does_IndAIF_exists(self):
        Old_IndAIF_path = os.path.join(self.patient_calc_path, "AIF/IndAIF_s{}_.npz".format(self.AIF_slice))
        IndAIF_path = os.path.join(self.patient_calc_path, "AIF/IndAIF_t{}_.npz".format(self.AIF_time))

        return os.path.exists(IndAIF_path)

    def load_IndAIF(self):
        Old_IndAIF_path = os.path.join(self.patient_calc_path, "AIF/IndAIF_s{}_.npz".format(self.AIF_slice))
        IndAIF_path = os.path.join(self.patient_calc_path, "AIF/IndAIF_t{}_.npz".format(self.AIF_time))
        AIF_data = np.load(IndAIF_path)
        self.S0 = AIF_data['S0']
        self.IndAIF = AIF_data['AIF']
        self.IndAIF_signal = AIF_data['AIF_signal']
        self.arteryROI = AIF_data['arteryROI']
        self.IndAIF_popt = AIF_data['popt']
        self.IndAIF_popt_std = AIF_data['std']
        self.IndAIF_T1 = AIF_data['T1']

    def get_IndAIF(self):
        Old_IndAIF_path = os.path.join(self.patient_calc_path, "AIF/IndAIF_s{}_.npz".format(self.AIF_slice))
        IndAIF_path = os.path.join(self.patient_calc_path, "AIF/IndAIF_t{}_.npz".format(self.AIF_time))
        AIF_data = np.load(IndAIF_path)
        return AIF_data['AIF']


    def calculate_IndAIF(self):
        # Calculate individual AIF
        Ca, Ca_signal, artery_ROI, S0, T1_mean = Modelling.IndAIF(self.patient_data_path, self.AIF_slice, 9, self.injection_time,
                                                         self.T1, self.R_Gd, self.TR, self.FA)

        # Can't do curvefitting if there exists nan values. Nan values are converted to 0 and inf to very large numbers
        Ca = np.nan_to_num(Ca)

        # Finding the curve following the AIF
        sigma1 = 5.
        sigma2 = 8.
        init_param = [np.max(Ca) * np.sqrt(2 * np.pi) * sigma1, np.max(Ca) * np.sqrt(2. * np.pi) * sigma2 / 7.,
                      5., 8., 35., 50., np.max(Ca) / 7, 0.003, 0.6, 50.]
        upper_bound = [np.max(Ca) * np.sqrt(2 * np.pi) * sigma1,
                       np.max(Ca) * np.sqrt(2. * np.pi) * sigma2 * 1.3 / 7., 7., 10., 50., 60., 0.1, 0.01, 10., 57.]
        lower_bound = [np.max(Ca) * np.sqrt(2 * np.pi) * sigma1 / 2.,
                       (np.max(Ca) * np.sqrt(2. * np.pi) * sigma2 / 7.) / 2., 3., 5., 5., 30., 0.0001, 0.0001, 0.01,
                       40.]

        time = np.arange(0, 60 * 3.735, 3.735)
        popt, pcov = curve_fit(Modelling.population_AIF_func, time, Ca, init_param, bounds=(lower_bound, upper_bound), max_nfev=100000)
        std = np.sqrt(np.diag(pcov))

        # SAVE DATA
        AIF_dir_path = os.path.join(self.patient_calc_path, "AIF")
        if os.path.exists(AIF_dir_path) == False:
            os.makedirs(AIF_dir_path)

        #self.IndAIF_path = os.path.join(AIF_dir_path, "IndAIF_s{}_.npz".format(self.AIF_slice))
        self.IndAIF_path = os.path.join(AIF_dir_path, "IndAIF_t{}_.npz".format(self.AIF_time))
        np.savez(self.IndAIF_path, S0=S0, T1=T1_mean, AIF=Ca, AIF_signal=Ca_signal, arteryROI=artery_ROI, popt=popt, std=std)
        print("IndAIF for patient {} has been saved to {}".format(self.patient_number, self.IndAIF_path))


    def add_Brix_constants(self):
        found_dicom = False
        dicom_path = os.path.join(self.patient_data_path, "dicom")
        for file in os.listdir(dicom_path):
            if file[-3:] == "IMA":
                file_path = os.path.join(dicom_path, file)
                dicom_header = pd.dcmread(file_path)
                bolus_volume = dicom_header[0x0018, 0x1041].value
                bolus_volume = float(bolus_volume)  # ml
                injection_speed = self.injection_speed  # ml/s
                tau = bolus_volume / injection_speed
                if self.sek_or_min == 'm' or self.sek_or_min == 'm' or self.sek_or_min == 'min' or self.sek_or_min == 'Min':
                    tau = tau/60.
                self.tau = tau
                found_dicom = True
                break

        if found_dicom == False:
            raise Exception('No bolus volume was retrieved because there was no dicom in {}'.format(dicom_path))

    def add_tumours(self):
        if hasattr(self, "chosen_tumour_segmentations") == False:
            tumour_segmentation_folder = os.path.join(self.patient_data_path, "nifti/dce_segmented/Tumours")
            chosen_tumour_segmentations = []
            chosen_tumour_names = []
            for tumour_name in os.listdir(tumour_segmentation_folder):
                if tumour_name[:4] == 'mask':
                    tumour_path = os.path.join(tumour_segmentation_folder, tumour_name)
                    tumour_segmentation = Modelling.get_image_numpy_data(tumour_path)
                    chosen_tumour_segmentations.append(tumour_segmentation)
                    chosen_tumour_names.append( tumour_name[5:-7])
            self.chosen_tumour_segmentations = chosen_tumour_segmentations
            self.chosen_tumour_names = chosen_tumour_names


    def calculate_pharmacokinetic_parameters(self, model, quant_model):
        # Add tumour segmentations to the Patient object
        self.add_tumours()

        # Get concentration, signal and affine
        if hasattr(self, "Ct") == False:
            if self.does_Ct_exist() == False:
                self.calculate_Ct(self.injection_time, self.R_Gd, self.TR, self.FA)

            else:
                self.load_Ct()

        parameters_dir_path = os.path.join(self.patient_calc_path, "Parameters/{}".format(model))
        if os.path.exists(parameters_dir_path) == False:
            os.makedirs(parameters_dir_path)

        for tumour_seg, tumour_name in zip(self.chosen_tumour_segmentations, self.chosen_tumour_names):
            print("Calculating parameters for tumour {}".format(tumour_name))

            if model == "Brix":
                # Calculating tau and adding it as an attribute to Patient if it does not exist
                if hasattr(self, "tau") == False:
                    self.add_Brix_constants()

                # Calculating pharmacokinetic parameters form Brix model
                Modelling.pharmacokinetic_parameter_calc(self.Ct, self.S, self.time, tumour_seg, model, tumour_name,
                                                parameters_dir_path, self.affine, self.injection_time, tau=self.tau)

                print("Calculated Brix parameters")

            else:
                # Individual AIF
                if 1 in quant_model['AIF_number']:
                    AIF_type = "IndAIF"
                    if self.does_IndAIF_exists() == False:
                        self.calculate_IndAIF()

                    AIF = self.get_IndAIF()

                    Modelling.pharmacokinetic_parameter_calc(self.Ct, self.S, self.time, tumour_seg, model, tumour_name,
                                                         parameters_dir_path, self.affine, self.injection_time, AIF=AIF,
                                                         AIF_type=AIF_type, artery_slice=self.AIF_slice,
                                                         artery_time=self.AIF_time)

                # Population AIF
                if 2 in quant_model['AIF_number']:
                    popAIF_type_list = ["PopAIF", "PopAIFrmBl", "PopAIFpa", "PopAIFpaRmBl", "PopAIFwia",
                                        "PopAIFwiaRmBl"]
                    popt_input_dict = {"PopAIF": "popt", "PopAIFrmBl": "popt_rm_bl",
                                       "PopAIFpa": "popt_aligned_peaks", "PopAIFpaRmBl": "popt_aligned_peaks_rm_bl",
                                       "PopAIFwia": "popt_aligned_wi", "PopAIFwiaRmBl": "popt_aligned_wi_rm_bl"}

                    for popAIF_num in quant_model['PopAIF_number']:
                        popAIF_path = os.path.join(self.calc_path, "PopAIF/{}".format(quant_model["PopAIFs"][popAIF_num-1]))
                        patients = popAIF_path.split("_")
                        patients = patients[-2]
                        patients = patients[1:]
                        popAIF_data = np.load(popAIF_path)
                        for popAIF_type in popAIF_type_list:
                            poptAIF_type = popt_input_dict[popAIF_type]
                            popAIF_popt = popAIF_data[poptAIF_type]
                            if popAIF_type[-2:] == "Bl":
                                chosen_time = self.time_rm_bl
                                chosen_conc = self.Ct
                                chosen_conc = chosen_conc[self.injection_time:, :, :, :]
                                chosen_signal = self.S
                                chosen_signal = chosen_signal[self.injection_time, :, :, :]

                            else:
                                chosen_time = self.time
                                chosen_conc = self.Ct
                                chosen_signal = self.S

                            if self.sek_or_min == 'm' or self.sek_or_min == "M":
                                popAIF_time = chosen_time*60.

                            AIF = Modelling.population_AIF_func(popAIF_time, *popAIF_popt)
                            Modelling.pharmacokinetic_parameter_calc(chosen_conc, chosen_signal, chosen_time, tumour_seg,
                                                                model, tumour_name, parameters_dir_path, self.affine,
                                                                self.injection_time, AIF=AIF, AIF_type=popAIF_type,
                                                                artery_slice=self.AIF_slice, AIF_patients=patients)

                print("Calculated Tofts parameters")


    def AUC_exists(self, AUC_values):
        AUC_dir = os.path.join(self.patient_calc_path, "AUC")
        existing_AUC_values = []
        if os.path.exists(AUC_dir):
            for AUC in os.listdir(AUC_dir):
                if AUC[:3] == "AUC" :
                    AUC_meta = AUC.split('_')
                    AUC_value = AUC_meta[0][3:]
                    existing_AUC_values.append(float(AUC_value))

            chosen_AUC_exists = []
            for i, AUC in enumerate(AUC_values):
                chosen_AUC_exists.append((int(AUC) in existing_AUC_values))

        else:
            chosen_AUC_exists = np.zeros(len(AUC_values), dtype=int)


        print(chosen_AUC_exists)
        return chosen_AUC_exists


    def calculate_AUC(self, AUC_time_values, TA, dt):
        self.add_tumours()
        if self.does_Ct_exist():
            self.load_Ct()

        else:
            self.calculate_Ct(self.injection_time, self.R_Gd, self.TR, self.FA)

        AUC_path = os.path.join(self.patient_calc_path, "AUC")
        if os.path.exists(AUC_path) == False:
            os.makedirs(AUC_path)

        S0 = np.mean(self.S[:self.injection_time, :, :, :], axis=0)
        CI = (self.S[TA:, :, :, :] - S0)/S0
        CI[CI < 0] = 0.0
        time = self.time[TA:]-self.time[TA]
        time = time*60.

        for i, (tumour_seg, tumour_name) in enumerate(zip(self.chosen_tumour_segmentations, self.chosen_tumour_names)):
            print("Calculating AUC for {}".format(tumour_name))
            ROI_AUC = np.zeros(len(AUC_time_values))
            for j, AUC_time in enumerate(AUC_time_values):
                print("Calculating AUC{}".format(AUC_time))
                ROI_AUC[j] = Modelling.AUC_2D_calculation(CI, time, tumour_seg, tumour_name,
                                                       AUC_time, dt, self.affine, AUC_path)

            ROI_AUC_path = os.path.join(AUC_path, "ROI_AUC_tumour_{}_.npz".format(tumour_name))
            np.savez(ROI_AUC_path, AUC=ROI_AUC, dt=dt, AUC_time=AUC_time_values, tumour_name=tumour_name)

    def TTHP_exists(self):
        """
        A function that checks if the time-to-half-peak has already been calculated for the patient.
        :return: a boolean, True if TTHP exists and False if it doesn't for all the patient's tumours
        """
        TTHP_dir = os.path.join(self.patient_calc_path, "TTHP")
        TTHP_exist = True
        if len(self.chosen_tumour_names) > 0:
            for tumour_name in self.chosen_tumour_names:
                TTHP_ROI_path = os.path.join(TTHP_dir, "ROI_TTHP_tumour_{}_.npz".format(tumour_name))
                TTHP_map_path = os.path.join(TTHP_dir, "TTHP_tumour_{}_.nii.gz".format(tumour_name))
                if os.path.exists(TTHP_ROI_path) == False or os.path.exists(TTHP_map_path) == False:
                    TTHP_exist = False
                    break
        else:
            TTHP_exist = False

        return TTHP_exist

    def calculate_TTHP(self):
        # Add the patient's tumour to the database. The tumour segmentations then become available
        self.add_tumours()

        # Load Ct in order to get the signal
        if self.does_Ct_exist():
            self.load_Ct()

        else:
            self.calculate_Ct(self.injection_time, self.R_Gd, self.TR, self.FA)

        # Remove the baseline from the signal
        S_rm_bl = self.S[self.injection_time:, :, :, :]

        # Create a folder to store TTHP data if such a folder does not already exist
        TTHP_dir_path = os.path.join(self.patient_calc_path, "TTHP")
        if os.path.exists(TTHP_dir_path) == False:
            os.makedirs(TTHP_dir_path)

        # Calculate the TTHP for each tumour in the patient. Both ROI and v-by-v is calculated.
        for (tumour_seg, tumour_name) in zip(self.chosen_tumour_segmentations, self.chosen_tumour_names):
            Modelling.TTHP_calculation(S_rm_bl, self.time_rm_bl, tumour_seg, tumour_name, TTHP_dir_path, self.affine)

    def retrieve_parameters(self, chosen_parameters, primary_or_nodal):
        conversion_scalar = {'A': 3600., 'Kel': 60, 'Kep': 60, 'Kt': 60, 've': 1, 'vp': 1, 'AUC': 1}
        parameters = []
        mean_parameters = []
        ROI_parameters = []
        median_parameters = []
        for tumour_mask, tumour_name in zip(self.chosen_tumour_segmentations, self.chosen_tumour_names):
            if tumour_name[:4] == "GTV{}".format(primary_or_nodal):
                tumour_parameters = []
                mean_tumour_parameters = []
                median_tumour_parameters = []
                ROI_tumour_parameters = []
                for parameter_meta in chosen_parameters:
                    parameter = parameter_meta.split("_")[0]
                    try:
                        model = parameter_meta.split("_")[1]
                    except:
                        pass
                    if parameter_meta[:3] == "AUC":
                        parameter_path = os.path.join(self.patient_calc_path,
                                                      "AUC/{}_tumour_{}_.nii.gz".format(parameter_meta, tumour_name))
                        parameter = parameter_meta[:3]

                    elif model == 'Brix':
                        parameter_path = os.path.join(self.patient_calc_path, "Parameters/{}/{}_model_{}_"
                                    "tumour_{}_.nii.gz".format(model, parameter, model, tumour_name))

                    else:
                        parameter_path = os.path.join(self.patient_calc_path, "Parameters/{}/{}_IndAIF_t{}_model_{}_"
                                    "tumour_{}_.nii.gz".format(model, parameter, self.AIF_time, model, tumour_name))



                    # Finding the indices for where in the parameter map the tumour is
                    tumour_mask_flat = tumour_mask.flatten()
                    tumour_idxs = np.where(tumour_mask_flat == True)

                    # Retrieving the parameter map
                    parameter_map = Modelling.get_image_numpy_data(parameter_path)
                    parameter_map_flat = parameter_map.flatten()
                    parameter_segmented = parameter_map_flat[tumour_idxs]
                    #scalar = conversion_scalar[parameter]
                    #parameter_segmented = parameter_segmented*scalar

                    # Calculate the mean of the parameters in the tumour
                    mean_parameter = np.mean(parameter_segmented)
                    mean_tumour_parameters = np.append(mean_tumour_parameters, mean_parameter)

                    # Calculate the median of the parameters in the tumour
                    median_parameter = np.median(parameter_segmented)
                    median_tumour_parameters = np.append(median_tumour_parameters, median_parameter)


                    if tumour_parameters == []:
                        tumour_parameters = parameter_segmented
                        tumour_parameters = np.reshape(tumour_parameters, (len(tumour_parameters), 1))

                    else:
                        tumour_parameters = np.concatenate((tumour_parameters, np.array([parameter_segmented]).T), axis=1)

                if parameters == []:
                    parameters = tumour_parameters
                else:
                    parameters = np.concatenate((parameters, tumour_parameters), axis=0)

                if mean_parameters == []:
                    mean_parameters = np.array([mean_tumour_parameters])
                else:
                    mean_parameters = np.concatenate((mean_parameters, np.array([mean_tumour_parameters])), axis=0)

                if median_parameters == []:
                    median_parameters = np.array([median_tumour_parameters])
                else:
                    median_parameters = np.concatenate((median_parameters, np.array([median_tumour_parameters])), axis=0)

                ROI_param_filenames = ["Parameters/Brix/MeanROI_param_model_Brix_tumour_{}_rm_bl_.npz".format(tumour_name),
                                       "Parameters/TM/MeanROI_param_IndAIF_t{}_model_TM_tumour_{}_.npz".format(self.AIF_time, tumour_name),
                                       "Parameters/ETM/MeanROI_param_IndAIF_t{}_model_ETM_tumour_{}_.npz".format(self.AIF_time, tumour_name),
                                       "AUC/ROI_AUC_tumour_{}_.npz".format(tumour_name)]

                for ROI_param_filename in ROI_param_filenames:
                    ROI_param_path = os.path.join(self.patient_calc_path, ROI_param_filename)
                    ROI_param_data = np.load(ROI_param_path)
                    if ROI_param_filename[:3] == "AUC":
                        ROI_param = ROI_param_data['AUC']
                    else:
                        ROI_param = ROI_param_data['ROI_param'].flatten()

                    metadata = ROI_param_filename.split("/")
                    model = metadata[1]
                    if model == "TM" or model == "ETM":
                        kep = ROI_param[0]/ROI_param[1]
                        ROI_param = np.append(ROI_param, kep)

                    if ROI_tumour_parameters == []:
                        ROI_tumour_parameters = ROI_param
                    else:
                        ROI_tumour_parameters = np.concatenate((ROI_tumour_parameters, ROI_param), axis=0)


                if ROI_parameters == []:
                    ROI_parameters = np.array([ROI_tumour_parameters])

                else:
                    ROI_parameters = np.concatenate((ROI_parameters, np.array([ROI_tumour_parameters])), axis=0)


        return parameters, ROI_parameters, mean_parameters, median_parameters

    def all_parameters_exist(self):
        models = ['Brix', 'TM', 'ETM', 'AUC']
        parameters = {'Brix': ['A', 'Kep', 'Kel'], 'TM': ['Kt', 've', 'Kep'],
                      'ETM': ['Kt', 've', 'vp', 'Kep'], 'AUC': ['60', '90', '120']}

        missing_models = set([])
        for model in models:
            try:
                for parameter in parameters[model]:
                    for tumour in self.chosen_tumour_names:
                        # Checking parameter maps
                        if model == 'AUC':
                                parameter_path = os.path.join(self.patient_calc_path, "AUC/AUC{}_tumour_"
                                                              "{}_.nii.gz".format(parameter, tumour))

                        elif model == 'Brix':
                            parameter_path = os.path.join(self.patient_calc_path, "Parameters/{}/{}_model_{}_tumour_"
                                                          "{}_.nii.gz".format(model, parameter, model, tumour))

                        else:
                            parameter_path = os.path.join(self.patient_calc_path, "Parameters/{}/{}_IndAIF_t{}_model_{}_"
                                             "tumour_{}_.nii.gz".format(model, parameter, self.AIF_time, model, tumour))

                        if os.path.exists(parameter_path) == False:
                            missing_models.add(model)
                            raise Missing

                        # Checking ROI values
                        if model == "AUC":
                            parameter_path = os.path.join(self.patient_calc_path, "AUC/ROI_AUC_tumour_{}_.npz".format(tumour))
                        elif model == "Brix":
                            parameter_path = os.path.join(self.patient_calc_path, "Parameters/{}/MeanROI_param_model_{}_"
                                             "tumour_{}_rm_bl_.npz".format(model, model, tumour))
                        else:
                            parameter_path = os.path.join(self.patient_calc_path, "Parameters/{}/MeanROI_param_IndAIF_t{}_"
                                             "model_{}_tumour_{}_.npz".format(model, self.AIF_time, model, tumour))
                        if os.path.exists((parameter_path)) == False:
                            missing_models.add(model)
                            raise Missing

            except Missing:
                continue

        if missing_models == set([]):
            return True, missing_models

        else:
            return False, missing_models


class Missing(Exception): pass













