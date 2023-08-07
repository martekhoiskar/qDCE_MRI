import pandas as pd
from math import isnan
import numpy as np
import os
def get_TNM_data(file):

    # Import CSV data to a Pandas DataFrame
    df = pd.read_csv(file, delimiter=";", header=None, index_col=0)
    dict = df.to_dict()
    print(dict)

    # Create dictionary for the different stages
    T_negative = dict[1]
    T_positive = dict[2]
    N_negative = dict[3]
    N_positive = dict[4]

    T_negative.pop('Patient number')
    T_positive.pop('Patient number')
    N_negative.pop('Patient number')
    N_positive.pop('Patient number')

    # Remove patients that have nan values
    T_negative = {k: T_negative[k] for k in T_negative if not isnan(float(T_negative[k]))}
    T_positive = {k: T_positive[k] for k in T_positive if not isnan(float(T_positive[k]))}
    N_negative = {k: N_negative[k] for k in N_negative if not isnan(float(N_negative[k]))}
    N_positive = {k: N_positive[k] for k in N_positive if not isnan(float(N_positive[k]))}

    # Create dictionaries that include data for both positive and negative HPV patients
    T_nANDp = T_negative | T_positive
    N_nANDp = N_negative | N_positive

    return T_negative, T_positive, T_nANDp, N_negative, N_positive, N_nANDp


def save_T_stage_data(ROI_values_list, TNM_folder, param, HPV):

    no_of_T1 = len(ROI_values_list[0])
    no_of_T2 = len(ROI_values_list[1])
    no_of_T3 = len(ROI_values_list[2])
    if len(ROI_values_list) == 4:
        no_of_T4 = len(ROI_values_list[3])
        T4_exists = True

    T1 = np.array(['T1']*no_of_T1)
    T2 = np.array(['T2']*no_of_T2)
    T3 = np.array(['T3']*no_of_T3)
    if T4_exists:
        T4 = np.array(['T4']*no_of_T4)

    T1_T2 = np.append(T1, T2)
    T1_T2_ROI_values = np.append(ROI_values_list[0], ROI_values_list[1])

    T1_T3 = np.append(T1, T3)
    T1_T3_ROI_values = np.append(ROI_values_list[0], ROI_values_list[2])

    if T4_exists:
        T1_T4 = np.append(T1, T4)
        T1_T4_ROI_values = np.append(ROI_values_list[0], ROI_values_list[3])
        T_stage_excel_list = [T1_T2, T1_T3, T1_T4]
        ROI_excel_list = [T1_T2_ROI_values, T1_T3_ROI_values, T1_T4_ROI_values]

    else:
        T_stage_excel_list = [T1_T2, T1_T3]
        ROI_excel_list = [T1_T2_ROI_values, T1_T3_ROI_values]

    T_stage_headers = ["T1vsT2", "T1vsT3", "T1vsT4"]
    ROI_headers = ["ROI_T1vsT2", "ROI_T1vsT3", "ROI_T1vsT4"]

    for i in range(len(T_stage_excel_list)):
        TNM_param_folder = os.path.join(TNM_folder, "{}".format(param))
        if os.path.exists(TNM_param_folder) == False:
            os.makedirs(TNM_param_folder)
        excel_path = os.path.join(TNM_param_folder, "{}_{}_{}.xlsx".format(param, T_stage_headers[i], HPV))
        df = pd.DataFrame([T_stage_excel_list[i], ROI_excel_list[i]])
        df = df.transpose()
        df.to_excel(excel_path, header=[T_stage_headers[i], ROI_headers[i]], index=False)




