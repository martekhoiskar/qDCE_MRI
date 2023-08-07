################################################################################
######################## I M P O R T  P A C K A G E S ##########################
################################################################################
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
from scipy.optimize import curve_fit
from src import pharmacokinetic_modelling as pmod
from scipy.special import betainc
import pandas as pd


################################################################################
####### S I G N A L  A N D  C O N C E N T R A T I O N  F U N C T I O N #########
################################################################################

def get_image_numpy_data(file):
    # Convert the MRI nifti file to numpy array
    img = nib.load(file)
    img_data = img.get_fdata()
    img_data = np.float64(img_data)
    return img_data


def get_signal_data(dir_loc, timepoints=60, cols=288, rows=320, slices=22):
    # Get folder with dce images
    dce_dir = os.path.join(dir_loc, "nifti/dce/")

    # Return a numpy array of the signal from a MRI slice at each time point
    signal = np.zeros((timepoints, cols, rows, slices))
    sorted_dir = sorted(os.listdir(dce_dir))
    dce_files = []
    for i, file in enumerate(sorted_dir):
        if file[:2] != "._" and file[-3:] != "csv":
            dce_files.append(file)

    for i, file in enumerate(sorted(dce_files)):
        file_path = os.path.join(dce_dir, file)
        img = nib.load(file_path)
        affine = img.affine
        img_data = img.get_fdata()
        img_data = np.float64(img_data)
        signal[i,:,:,:] = img_data[:,:,:]
    return signal, affine


def signal_to_conc(T1map, S, S0, R_Gd, TR=3.04, FA=12):
    # Calculates the tissue concentration over time for each pixel
    # in a slice from the raw MRI data using the approach from QIBA
    FA_rad = FA*np.pi/180. #Convert from degrees to radians
    E10 = np.exp(-TR/T1map)
    B = (1.-E10)/(1.-E10*np.cos(FA_rad))
    S_times_B = np.multiply(S, B)
    A = np.divide(S_times_B, S0)
    R1 = -1./TR*np.log((1.-A)/(1.-np.cos(FA_rad)*A))
    conc = (R1-1./T1map)/R_Gd
    return conc


def tumour_conc_calc(data_dcm, injection_time, R_Gd, TR, FA):
    # SIGNAL
    signal, affine = get_signal_data(data_dcm)

    # TUMOUR CONCENTRATION
    # T1 map
    T1map_loc = os.path.join(data_dcm, "nifti/t1_images_b1corr_res2dce.nii.gz")
    T1map = get_image_numpy_data(T1map_loc)

    # Baseline
    S0 = np.mean(signal.copy()[0:injection_time, :, :, :], axis=0)

    # Concentration calculation
    conc = signal_to_conc(T1map, signal, S0, R_Gd, TR, FA)

    return signal, conc, affine

def IndAIF(data_dcm, artery_slice, artery_timepoint, injection_time, T1, R_Gd, TR, FA):
    # SIGNAL
    signal, affine = get_signal_data(data_dcm)
    nt, cols, rows, slices = signal.shape


    # AIF CONCENTRATION
    # Find artery ROI map
    arterial_ROI_path = os.path.join(data_dcm, "nifti/dce_segmented/Arteries/ArterySegmentation.nii.gz")
    artery_ROI_map = get_image_numpy_data(arterial_ROI_path)

    # Find artery ROI indices
    artery_ROI_map_flattened = artery_ROI_map.copy()
    artery_ROI_map_flattened = artery_ROI_map_flattened.flatten()
    artery_ROI_indices = np.where(artery_ROI_map_flattened == True)

    # Find the 5 % voxels with the highest value
    temp_artery_ROI_signal = signal.copy()
    temp_artery_ROI_signal = temp_artery_ROI_signal[artery_timepoint, :, :, :]
    temp_artery_ROI_signal = temp_artery_ROI_signal.flatten()
    temp_artery_ROI_signal = temp_artery_ROI_signal[artery_ROI_indices]

    number_of_voxels = len(artery_ROI_indices[0])
    five_percent = int(0.05 * number_of_voxels)
    high_ind = np.argsort(temp_artery_ROI_signal)[-five_percent:]

    # Find T1
    T1_path = os.path.join(data_dcm, "nifti/t1_images_b1corr_res2dce.nii.gz")
    T1_map = get_image_numpy_data(T1_path)
    T1_map_flattened = T1_map.flatten()
    T1_map_artery = T1_map_flattened[artery_ROI_indices]
    T1_map_artery = T1_map_artery[high_ind]
    T1_mean = np.mean(T1_map_artery)

    artery_time_signal = np.zeros(nt)
    for t in range(nt):
        # Signal in artery
        artery_ROI_signal = signal.copy()
        artery_ROI_signal = artery_ROI_signal[t, :, :, :]
        artery_ROI_signal = artery_ROI_signal.flatten()
        artery_ROI_signal = artery_ROI_signal[artery_ROI_indices]

        # 5 % voxels with the highest value
        artery_ROI_signal = artery_ROI_signal[high_ind]

        # Average signal
        artery_time_signal[t] = np.mean(artery_ROI_signal)

    # Baseline
    S0 = np.mean(artery_time_signal[:injection_time])

    # Concentration calculation
    artery_time_concentration = signal_to_conc(T1_mean, artery_time_signal, S0, R_Gd, TR, FA)

    time = np.arange(0, 60*3.735, 3.735)
    plt.figure()
    plt.plot(time, artery_time_concentration)
    plt.title(data_dcm)
    plt.show()
    return artery_time_concentration, artery_time_signal, artery_ROI_map, S0, T1_mean

def population_AIF_func(time, a1, a2, sigma1, sigma2, T1, T2, alpha, beta, s, tau):
    """
    Population AIF as a function of time given by Parker et. al (2006).
    :param time:
    :param a1:
    :param a2:
    :param sigma1:
    :param sigma2:
    :param T1:
    :param T2:
    :param alpha:
    :param beta:
    :param s:
    :param tau:
    :return:
    """
    term1 = a1*np.exp(-(time-T1)**2/(2*sigma1**2))/(sigma1*np.sqrt(2*np.pi))
    term2 = a2*np.exp(-(time-T2)**2/(2*sigma2**2))/(sigma2*np.sqrt(2*np.pi))
    term3 = alpha*np.exp(-beta*time)/(1+np.exp(-s*(time-tau)))
    Cb = term1 + term2 + term3
    return Cb

def align_peaks(AIF_array):
    max_indices = np.argmax(AIF_array, axis=1)
    new_maximum_idx = np.mean(max_indices)
    new_maximum_idx = round(new_maximum_idx)

    maximum_diff_idx = new_maximum_idx - max_indices
    aligned_AIF = np.zeros_like(AIF_array)

    for i, AIF in enumerate(AIF_array):
        max_diff = maximum_diff_idx[i]
        aligned_AIF[i, :] = np.roll(AIF, max_diff)
        if max_diff < 0:
            aligned_AIF[i, max_diff:] = np.array([np.nan for i in range(abs(max_diff))])
        elif max_diff > 0:
            aligned_AIF[i, :max_diff] = np.array([np.nan for i in range(max_diff)])

    return aligned_AIF, maximum_diff_idx, new_maximum_idx

def align_wash_in(AIF_array, limit=0.0002):
    wash_in_indx = np.sum(AIF_array[:, :15] < limit, axis=1)
    wash_in_indx = wash_in_indx - 1
    new_wash_in_indx = np.mean(wash_in_indx)
    new_wash_in_indx = round(new_wash_in_indx)

    wash_in_diff_indx = new_wash_in_indx - wash_in_indx
    aligned_AIF = np.zeros_like(AIF_array)

    for i, AIF in enumerate(AIF_array):
        wash_in_diff = wash_in_diff_indx[i]
        aligned_AIF[i, :] = np.roll(AIF, wash_in_diff)
        if wash_in_diff < 0:
            aligned_AIF[i, wash_in_diff:] = np.array([np.nan for i in range(abs(wash_in_diff))])
        elif wash_in_diff > 0:
            aligned_AIF[i, :wash_in_diff] = np.array([np.nan for i in range(wash_in_diff)])

    return aligned_AIF, wash_in_diff_indx, new_wash_in_indx


def calculate_popAIFs(indAIF_list, time, time_rm_bl, injection_time, R_Gd, TR, FA, T1):

    sigma1 = 5
    sigma2 = 10

    # No alignment and baseline included
    popAIF_signal = np.mean(indAIF_list, axis=0)
    S0 = np.mean(popAIF_signal[:injection_time])
    popAIF = signal_to_conc(T1, popAIF_signal, S0, R_Gd, TR, FA)

    init_param = [np.max(popAIF) * np.sqrt(2 * np.pi) * sigma1,
                  np.max(popAIF) * np.sqrt(2 * np.pi) * sigma2 / 5,
                  sigma1, sigma2, 35, 60, 0.002, 0.003, 0.5, 50]
    upper_bound = [np.max(popAIF) * np.sqrt(2 * np.pi) * sigma1 * 1.3,
                   np.max(popAIF) * np.sqrt(2 * np.pi) * sigma2 * 1.3 / 5,
                   10, 20, 50, 70, 0.1, 0.01, 1, 60]
    lower_bound = [np.max(popAIF) * np.sqrt(2 * np.pi) * sigma1 / 2,
                   (np.max(popAIF) * np.sqrt(2 * np.pi) * sigma2 / 5) / 2,
                   1, 1, 20, 40, 0.0001, 0.0001, 0.0, 40]

    # No alignment and baseline removed
    popAIF_rm_bl = popAIF[injection_time:]

    init_param_rm_bl = [np.max(popAIF_rm_bl) * np.sqrt(2 * np.pi) * sigma1,
                  np.max(popAIF_rm_bl) * np.sqrt(2 * np.pi) * sigma2 / 5,
                  sigma1, sigma2, 35, 60, 0.002, 0.003, 0.5, 50]
    upper_bound_rm_bl = [np.max(popAIF_rm_bl) * np.sqrt(2 * np.pi) * sigma1 * 1.3,
                   np.max(popAIF_rm_bl) * np.sqrt(2 * np.pi) * sigma2 * 1.3 / 5,
                   10, 20, 50, 70, 0.1, 0.01, 1, 60]
    lower_bound_rm_bl = [np.max(popAIF_rm_bl) * np.sqrt(2 * np.pi) * sigma1 / 2,
                   (np.max(popAIF_rm_bl) * np.sqrt(2 * np.pi) * sigma2 / 5) / 2,
                   1, 1, 20, 40, 0.0001, 0.0001, 0.0, 40]

    # Aligning peaks and include baseline
    indAIF_aligned_peaks, peak_indx, new_peak_indx = align_peaks(indAIF_list)
    popAIF_aligned_peaks_signal = np.nanmean(indAIF_aligned_peaks, axis=0)
    S0 = np.mean(popAIF_aligned_peaks_signal[:injection_time])
    popAIF_aligned_peaks = signal_to_conc(T1, popAIF_aligned_peaks_signal, S0, R_Gd, TR, FA)

    init_param_aligned_peaks = [np.max(popAIF_aligned_peaks) * np.sqrt(2 * np.pi) * sigma1,
                  np.max(popAIF_aligned_peaks) * np.sqrt(2 * np.pi) * sigma2 / 5,
                  sigma1, sigma2, 35, 60, 0.002, 0.003, 0.5, 50]
    upper_bound_aligned_peaks = [np.max(popAIF_aligned_peaks) * np.sqrt(2 * np.pi) * sigma1 * 1.3,
                   np.max(popAIF_aligned_peaks) * np.sqrt(2 * np.pi) * sigma2 * 1.3 / 5,
                   10, 20, 50, 70, 0.1, 0.01, 1, 60]
    lower_bound_aligned_peaks = [np.max(popAIF_aligned_peaks) * np.sqrt(2 * np.pi) * sigma1 / 2,
                   (np.max(popAIF_aligned_peaks) * np.sqrt(2 * np.pi) * sigma2 / 5) / 2,
                   1, 1, 20, 40, 0.0001, 0.0001, 0.0, 40]

    # Aligning peaks and remove baseline
    popAIF_aligned_peaks_rm_bl = popAIF_aligned_peaks[injection_time:]

    init_param_aligned_peaks_rm_bl = [np.max(popAIF_aligned_peaks_rm_bl) * np.sqrt(2 * np.pi) * sigma1,
                  np.max(popAIF_aligned_peaks_rm_bl) * np.sqrt(2 * np.pi) * sigma2 / 5,
                  sigma1, sigma2, 35, 60, 0.002, 0.003, 0.5, 45]
    upper_bound_aligned_peaks_rm_bl = [np.max(popAIF_aligned_peaks_rm_bl) * np.sqrt(2 * np.pi) * sigma1 * 1.3,
                   np.max(popAIF_aligned_peaks_rm_bl) * np.sqrt(2 * np.pi) * sigma2 * 1.3 / 5,
                   10, 20, 50, 70, 0.1, 0.01, 1, 60]
    lower_bound_aligned_peaks_rm_bl = [np.max(popAIF_aligned_peaks_rm_bl) * np.sqrt(2 * np.pi) * sigma1 / 2,
                   (np.max(popAIF_aligned_peaks_rm_bl) * np.sqrt(2 * np.pi) * sigma2 / 5) / 2,
                   1, 1, 20, 40, 0.0001, 0.0001, 0.0, 40]

    # Aligning wash in and include baseline
    indAIF_aligned_wi, wi_indx, new_wi_indx = align_wash_in(indAIF_list, limit=200)
    popAIF_aligned_wi_signal = np.nanmean(indAIF_aligned_wi, axis=0)
    S0 = np.mean(popAIF_aligned_wi_signal[:injection_time])
    popAIF_aligned_wi = signal_to_conc(T1, popAIF_aligned_wi_signal, S0, R_Gd, TR, FA)

    init_param_aligned_wi = [np.max(popAIF_aligned_wi) * np.sqrt(2 * np.pi) * sigma1,
                  np.max(popAIF_aligned_wi) * np.sqrt(2 * np.pi) * sigma2 / 5,
                  sigma1, sigma2, 35, 60, 0.002, 0.003, 0.5, 50]
    upper_bound_aligned_wi = [np.max(popAIF_aligned_wi) * np.sqrt(2 * np.pi) * sigma1 * 1.3,
                   np.max(popAIF_aligned_wi) * np.sqrt(2 * np.pi) * sigma2 * 1.3 / 5,
                   10, 20, 50, 70, 0.1, 0.01, 1, 60]
    lower_bound_aligned_wi = [np.max(popAIF_aligned_wi) * np.sqrt(2 * np.pi) * sigma1 / 2,
                   (np.max(popAIF_aligned_wi) * np.sqrt(2 * np.pi) * sigma2 / 5) / 2,
                   1, 1, 20, 40, 0.0001, 0.0001, 0.0, 40]

    # Aligning wash in and no baseline
    popAIF_aligned_wi_rm_bl = popAIF_aligned_wi[injection_time:]

    init_param_aligned_wi_rm_bl = [np.max(popAIF_aligned_wi_rm_bl) * np.sqrt(2 * np.pi) * sigma1,
                  np.max(popAIF_aligned_wi_rm_bl) * np.sqrt(2 * np.pi) * sigma2 / 5,
                  sigma1, sigma2, 35, 60, 0.002, 0.003, 0.5, 50]
    upper_bound_aligned_wi_rm_bl = [np.max(popAIF_aligned_wi_rm_bl) * np.sqrt(2 * np.pi) * sigma1 * 1.3,
                   np.max(popAIF_aligned_wi_rm_bl) * np.sqrt(2 * np.pi) * sigma2 * 1.3 / 5,
                   10, 20, 50, 70, 0.1, 0.01, 1, 60]
    lower_bound_aligned_wi_rm_bl = [np.max(popAIF_aligned_wi_rm_bl) * np.sqrt(2 * np.pi) * sigma1 / 2,
                   (np.max(popAIF_aligned_wi_rm_bl) * np.sqrt(2 * np.pi) * sigma2 / 5) / 2,
                   1, 1, 20, 40, 0.0001, 0.0001, 0.0, 40]

    # Grouping the AIF data together to do the curve-fit in a for-loop
    popAIF_list = [popAIF, popAIF_rm_bl, popAIF_aligned_peaks, popAIF_aligned_peaks_rm_bl,
                   popAIF_aligned_wi, popAIF_aligned_wi_rm_bl]
    time_list = [time, time_rm_bl, time, time_rm_bl, time, time_rm_bl]
    init_param_list = [init_param, init_param_rm_bl, init_param_aligned_peaks, init_param_aligned_peaks_rm_bl,
                                init_param_aligned_wi, init_param_aligned_wi_rm_bl]
    upperbound_list = [upper_bound, upper_bound_rm_bl, upper_bound_aligned_peaks, upper_bound_aligned_peaks_rm_bl,
                       upper_bound_aligned_wi, upper_bound_aligned_wi_rm_bl]
    lowerbound_list = [lower_bound, lower_bound_rm_bl, lower_bound_aligned_peaks, lower_bound_aligned_peaks_rm_bl,
                       lower_bound_aligned_wi, lower_bound_aligned_wi_rm_bl]

    print_list = ["\nPopt for pop-AIF with no alignement", "\nPopt for pop-AIF with no alignement and no baseline",
                  "\nPopt for pop-AIF with peak alignement", "\nPopt for pop-AIF with peak alignement and no baseline",
                  "\nPopt for pop-AIF with wash-in alignement",
                  "\nPopt for pop-AIF with wash-in alignement and no baseline"]

    popt_list = np.zeros((6, 10))
    std_list = np.zeros((6, 10))
    for i in range(6):

        popt_list[i, :], pcov = curve_fit(population_AIF_func, time_list[i], popAIF_list[i], init_param_list[i], bounds=(lowerbound_list[i], upperbound_list[i]), max_nfev=10000)
        std_list[i, :] = np.sqrt(np.diag(pcov))

        plt.figure()
        plt.plot(time_list[i], population_AIF_func(time_list[i], *popt_list[i, :]), color='gray')
        plt.scatter(time_list[i], popAIF_list[i], color='black')
        plt.show()

        print(print_list[i])
        print("A1:", popt_list[i, 0] * 1000 / 60, "+/-", std_list[i, 0] * 1000 / 60)
        print("A2:", popt_list[i, 1] * 1000 / 60, "+/-", std_list[i, 1] * 1000 / 60)
        print("Sigma1:", popt_list[i, 2] / 60, "+/-", std_list[i, 2] / 60)
        print("Sigma2:", popt_list[i, 3] / 60, "+/-", std_list[i, 3] / 60)
        print("T1:", popt_list[i, 4] / 60, "+/-", std_list[i, 4] / 60)
        print("T2:", popt_list[i, 5] / 60, "+/-", std_list[i, 5] / 60)
        print("alpha:", popt_list[i, 6] * 1000, "+/-", std_list[i, 6] * 1000)
        print("beta:", popt_list[i, 7] * 60, "+/-", std_list[i, 7] * 60)
        print("s:", popt_list[i, 8] * 60, "+/-", std_list[i, 8] * 60)
        print("tau:", popt_list[i, 9] / 60, "+/-", std_list[i, 9] / 60)

    return popAIF_list, popt_list, std_list, new_peak_indx, new_wi_indx


def pharmacokinetic_parameter_calc(Ct, S, time, tumour_seg, model, tumour_name, parameters_loc,
                                   affine, injection_time, **kwargs):

    if model == 'TM' or model == 'ETM':
        AIF_type = kwargs['AIF_type']
        artery_slice = kwargs['artery_slice']
        Cp = kwargs['AIF']
        if AIF_type == "IndAIF":
            artery_time = kwargs['artery_time']
            #AIF_name = AIF_type + "_s{}".format(artery_slice)
            AIF_name = AIF_type + "_t{}".format(artery_time)
        else:
            AIF_name = AIF_type + "_p{}".format(kwargs['AIF_patients'])

    nt, nx, ny, nz = Ct.shape

    # SLICES WITH TUMOUR
    tumour_pixels_in_slice = np.sum(tumour_seg, axis=(0, 1))
    tumour_slices = np.where(tumour_pixels_in_slice > 0)

    if model == "TM":


        Kt_map = np.zeros((nx, ny, nz))
        Kt_std_map = np.zeros((nx, ny, nz))
        ve_map = np.zeros((nx, ny, nz))
        ve_std_map = np.zeros((nx, ny, nz))
        R_sq_map = np.zeros((nx, ny, nz))

        for i, s_num in enumerate(tumour_slices[0]):
            print("Slice number:", s_num, "slice {} out of {}".format(i, len(tumour_slices[0])))
            tumour_mask = tumour_seg[:, :, s_num]
            Ct_slice = Ct.copy()[:, :, :, s_num]
            Kt_map[:, :, s_num], ve_map[:, :, s_num], Kt_std_map[:, :, s_num], ve_std_map[:, :,s_num], \
            R_sq_map[:, :, s_num], ROI_idxs = pmod.parameter_maps(Ct_slice, time, tumour_mask, model, Cp=Cp)

        Kt_path = os.path.join(parameters_loc, "Kt_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(Kt_map, affine=affine)
        nib.save(nifti_img, Kt_path)

        Kt_std_path = os.path.join(parameters_loc,
                                   "Kt_std_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(Kt_std_map, affine=affine)
        nib.save(nifti_img, Kt_std_path)

        ve_path = os.path.join(parameters_loc, "ve_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(ve_map, affine=affine)
        nib.save(nifti_img, ve_path)

        ve_std_path = os.path.join(parameters_loc,
                                   "ve_std_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(ve_std_map, affine=affine)
        nib.save(nifti_img, ve_std_path)

        Kep_map = Kt_map / ve_map
        Kep_std_map = np.sqrt((Kt_std_map / ve_map) ** 2 + (Kt_map * ve_std_map / (ve_map) ** 2) ** 2)

        Kep_path = os.path.join(parameters_loc,
                                "Kep_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(Kep_map, affine=affine)
        nib.save(nifti_img, Kep_path)

        Kep_std_path = os.path.join(parameters_loc,
                                "Kep_std_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(Kep_std_map, affine=affine)
        nib.save(nifti_img, Kep_std_path)


        R_sq_path = os.path.join(parameters_loc, "Rsq_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(R_sq_map, affine=affine)
        nib.save(nifti_img, R_sq_path)
        print("Saved Kt and ve and its std and Rsq in {}".format(parameters_loc))


        # PARAMETERS FOR MEAN TIC
        Ct_mean = np.zeros((len(time), len(tumour_slices[0])))
        print("Mean parameters calculations")
        for t in range(len(time)):
            for i, s_num in enumerate(tumour_slices[0]):
                tumour_mask = tumour_seg.copy()[:, :, s_num]
                tumour_mask_flat = tumour_mask.flatten()
                tumour_idxs = np.where(tumour_mask_flat == True)

                Ct_timepoint_slice = Ct.copy()[t, :, :, s_num]
                Ct_timepoint_slice = Ct_timepoint_slice.flatten()
                Ct_timepoint_slice_tumour = Ct_timepoint_slice[tumour_idxs]
                Ct_mean[t, i] = np.ma.masked_invalid(Ct_timepoint_slice_tumour).mean()


        # Parameters for mean of each slice
        mean_params_slices = np.zeros((len(tumour_slices[0]), 2))
        mean_std_slices = np.zeros((len(tumour_slices[0]), 2))
        """
        for s in range(len(tumour_slices[0])):
            mean_params_slices[s, :], mean_std_slices[s, :], Rsq = pmod.fit_tofts_model(np.array([Ct_mean[:, s]]), Cp,
                                                                                   time, extended=False)"""

        #start_time_list = [0]
        #start_time_name_list = ["_"]
        #start_time_list = [0, injection_time]
        #start_time_name_list = ["", "_rm_bl_"]

        start_time_list = [injection_time]
        start_time_name_list = ["rm_bl_"]
        for (start_time, time_name) in zip(start_time_list, start_time_name_list):
            Ct_ROI_mean = np.mean(Ct_mean, axis=1)
            chosen_time = time[start_time:]-time[start_time]
            mean_param, mean_stds, Rsq = pmod.fit_tofts_model(np.array([Ct_ROI_mean[start_time:]]), Cp[start_time:], chosen_time, extended=False)
            mean_param_path = os.path.join(parameters_loc,
                                           "MeanROI_param_{}_model_{}_tumour_{}_{}.npz".format(AIF_name, model,
                                                                                                tumour_name, time_name))
            np.savez(mean_param_path, Cp=Cp, Ct_mean=Ct_mean, ROI_param=mean_param, ROI_std=mean_stds, ROI_Rsq=Rsq,
                     mean_params_slices=mean_params_slices, mean_std_slices=mean_std_slices, slices=tumour_slices)


    elif model == "ETM":

        Kt_map = np.zeros((nx, ny, nz))
        Kt_std_map = np.zeros((nx, ny, nz))
        ve_map = np.zeros((nx, ny, nz))
        ve_std_map = np.zeros((nx, ny, nz))
        vp_map = np.zeros((nx, ny, nz))
        vp_std_map = np.zeros((nx, ny, nz))
        R_sq_map = np.zeros((nx, ny, nz))

        for i, s_num in enumerate(tumour_slices[0]):
            print("Slice number:", s_num, "slice {} out of {}".format(i, len(tumour_slices[0])))
            tumour_mask = tumour_seg[:, :, s_num]
            Ct_slice = Ct.copy()[:, :, :, s_num]
            Kt_map[:, :, s_num], ve_map[:, :, s_num], vp_map[:, :, s_num], Kt_std_map[:, :, s_num], \
            ve_std_map[:, :, s_num], vp_std_map[:, :, s_num], R_sq_map[:, :, s_num], \
            ROI_idxs = pmod.parameter_maps(Ct_slice, time, tumour_mask, model, Cp=Cp)


        Kep_map = Kt_map / ve_map
        Kep_std_map = np.sqrt((Kt_std_map / ve_map) ** 2 + (Kt_map * ve_std_map / (ve_map) ** 2) ** 2)

        Kt_path = os.path.join(parameters_loc, "Kt_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(Kt_map, affine=affine)
        nib.save(nifti_img, Kt_path)

        Kt_std_path = os.path.join(parameters_loc,
                                   "Kt_std_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(Kt_std_map, affine=affine)
        nib.save(nifti_img, Kt_std_path)

        ve_path = os.path.join(parameters_loc, "ve_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(ve_map, affine=affine)
        nib.save(nifti_img, ve_path)

        ve_std_path = os.path.join(parameters_loc,
                                   "ve_std_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(ve_std_map, affine=affine)
        nib.save(nifti_img, ve_std_path)

        vp_path = os.path.join(parameters_loc, "vp_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(vp_map, affine=affine)
        nib.save(nifti_img, vp_path)

        vp_std_path = os.path.join(parameters_loc,
                                   "vp_std_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(vp_std_map, affine=affine)
        nib.save(nifti_img, vp_std_path)

        Kep_path = os.path.join(parameters_loc,
                                "Kep_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(Kep_map, affine=affine)
        nib.save(nifti_img, Kep_path)

        Kep_std_path = os.path.join(parameters_loc,
                                    "Kep_std_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(Kep_std_map, affine=affine)
        nib.save(nifti_img, Kep_std_path)

        Rsq_path = os.path.join(parameters_loc,
                                    "Rsq_{}_model_{}_tumour_{}_.nii.gz".format(AIF_name, model, tumour_name))
        nifti_img = nib.Nifti1Image(R_sq_map, affine=affine)
        nib.save(nifti_img, Rsq_path)

        print("Saved Kt, ve, vp and kep and its std in {}".format(parameters_loc))

        # PARAMETERS FOR MEAN TIC
        Ct_mean = np.zeros((len(time), len(tumour_slices[0])))
        print("Mean parameters calculations")
        for t in range(len(time)):
            for i, s_num in enumerate(tumour_slices[0]):
                tumour_mask = tumour_seg.copy()[:, :, s_num]
                tumour_mask_flat = tumour_mask.flatten()
                tumour_idxs = np.where(tumour_mask_flat == True)

                Ct_timepoint_slice = Ct.copy()[t, :, :, s_num]
                Ct_timepoint_slice = Ct_timepoint_slice.flatten()
                Ct_timepoint_slice_tumour = Ct_timepoint_slice[tumour_idxs]
                Ct_mean[t, i] = np.ma.masked_invalid(Ct_timepoint_slice_tumour).mean()

        # Parameters for mean of each slice
        mean_params_slices = np.zeros((len(tumour_slices[0]), 3))
        mean_std_slices = np.zeros((len(tumour_slices[0]), 3))
        for s in range(len(tumour_slices[0])):
            mean_params_slices[s, :], mean_std_slices[s, :], Rsq = pmod.fit_tofts_model(np.array([Ct_mean[:, s]]), Cp,
                                                                                       time, extended=True)

        Ct_ROI_mean = np.mean(Ct_mean, axis=1)
        start_time_list = [0, injection_time]
        start_time_name_list = ["", "rm_bl_"]
        for (start_time, time_name) in zip(start_time_list, start_time_name_list):
            chosen_time = time[start_time:]-time[start_time]
            mean_param, mean_stds, Rsq = pmod.fit_tofts_model(np.array([Ct_ROI_mean[start_time:]]), Cp[start_time:], chosen_time, extended=True)

            mean_param_path = os.path.join(parameters_loc,
                                           "MeanROI_param_{}_model_{}_tumour_{}_{}.npz".format(AIF_name, model,
                                                                                                tumour_name, time_name))
            np.savez(mean_param_path, Cp=Cp, Ct_mean=Ct_mean, ROI_param=mean_param, ROI_std=mean_stds,  ROI_sq=Rsq,
                     mean_params_slices=mean_params_slices, mean_std_slices=mean_std_slices, slices=tumour_slices)

    else:


        Kep_map = np.zeros((nx, ny, nz))
        Kep_std_map = np.zeros((nx, ny, nz))
        Kel_map = np.zeros((nx, ny, nz))
        Kel_std_map = np.zeros((nx, ny, nz))
        A_map = np.zeros((nx, ny, nz))
        A_std_map = np.zeros((nx, ny, nz))
        R_sq_map = np.zeros((nx, ny, nz))

        tau = kwargs['tau']

        # Calculate S/S0
        S0 = S.copy()[:injection_time, :, :, :]
        S0 = np.mean(S0, axis=0)
        S_S0_ratio = S / S0

        # Baseline is removed because the Brix model is only defined for the infusion (injection of contrast)
        # and after infusion
        time_rm_bl = time[injection_time:] - time[injection_time]
        S_S0_ratio_rm_bl = S_S0_ratio[injection_time:, :, :, :]
        S_S0_ratio_rm_bl[0] = np.mean(S_S0_ratio[:injection_time+1, :, :, :], axis=0)

        # Calculate the parameters
        for i, s_num in enumerate(tumour_slices[0]):
            print("Slice number:", s_num, "(slice {} out of {})".format(i+1, len(tumour_slices[0])))
            tumour_mask = tumour_seg[:, :, s_num]
            S_S0_ratio_slice = S_S0_ratio_rm_bl.copy()[:, :, :, s_num]
            Kep_map[:, :, s_num], Kel_map[:, :, s_num], A_map[:, :, s_num], Kep_std_map[:, :, s_num], \
            Kel_std_map[:, :, s_num], A_std_map[:, :, s_num], \
            R_sq_map[:, :, s_num], ROI_idxs = pmod.parameter_maps(S_S0_ratio_slice, time_rm_bl, tumour_mask, model, tau=tau)


        Kep_path = os.path.join(parameters_loc, "Kep_model_{}_tumour_{}_.nii.gz".format(model, tumour_name))
        nifti_img = nib.Nifti1Image(Kep_map, affine=affine)
        nib.save(nifti_img, Kep_path)

        Kep_std_path = os.path.join(parameters_loc, "Kep_std_model_{}_tumour_{}_.nii.gz".format(model, tumour_name))
        nifti_img = nib.Nifti1Image(Kep_std_map, affine=affine)
        nib.save(nifti_img, Kep_std_path)

        Kel_path = os.path.join(parameters_loc, "Kel_model_{}_tumour_{}_.nii.gz".format(model, tumour_name))
        nifti_img = nib.Nifti1Image(Kel_map, affine=affine)
        nib.save(nifti_img, Kel_path)

        Kel_std_path = os.path.join(parameters_loc, "Kel_std_model_{}_tumour_{}_.nii.gz".format(model, tumour_name))
        nifti_img = nib.Nifti1Image(Kel_std_map, affine=affine)
        nib.save(nifti_img, Kel_std_path)

        A_path = os.path.join(parameters_loc, "A_model_{}_tumour_{}_.nii.gz".format(model, tumour_name))
        nifti_img = nib.Nifti1Image(A_map, affine=affine)
        nib.save(nifti_img, A_path)

        A_std_path = os.path.join(parameters_loc, "A_std_model_{}_tumour_{}_.nii.gz".format(model, tumour_name))
        nifti_img = nib.Nifti1Image(A_std_map, affine=affine)
        nib.save(nifti_img, A_std_path)

        R_sq_path = os.path.join(parameters_loc, "Rsq_model_{}_tumour_{}_.nii.gz".format(model, tumour_name))
        nifti_img = nib.Nifti1Image(R_sq_map, affine=affine)
        nib.save(nifti_img, R_sq_path)

        print("Saved Kep, Kel and A and its std and Rsq in {}".format(parameters_loc))


        # PARAMETERS FOR MEAN TIC
        S_S0_ratio_mean = np.zeros((len(time), len(tumour_slices[0])))
        print("Mean parameters calculations")
        for t in range(len(time)):
            for i, s_num in enumerate(tumour_slices[0]):
                tumour_mask = tumour_seg.copy()[:, :, s_num]
                tumour_mask_flat = tumour_mask.flatten()
                tumour_idxs = np.where(tumour_mask_flat == True)

                S_timepoint_slice = S.copy()[t, :, :, s_num]
                S_timepoint_slice = S_timepoint_slice.flatten()
                S_timepoint_slice_tumour = S_timepoint_slice[tumour_idxs]

                S0_slice = S.copy()[:injection_time, :, :, s_num]
                S0_slice = np.mean(S0_slice, axis=0)
                S0_slice = S0_slice.flatten()
                S0_slice_tumour = S0_slice[tumour_idxs]

                S_S0_ratio = S_timepoint_slice_tumour / S0_slice_tumour
                S_S0_ratio_mean[t, i] = np.ma.masked_invalid(S_S0_ratio).mean()

        # Remove baseline
        #S_S0_ratio_mean_rm_bl = S_S0_ratio_mean[7:, :]
        #S_S0_ratio_mean_rm_bl[0] = np.mean(S_S0_ratio_mean[:8, :], axis=0)
        S_S0_ratio_mean_rm_bl = S_S0_ratio_mean[injection_time:, :]
        S_S0_ratio_mean_rm_bl[0] = np.mean(S_S0_ratio_mean[:injection_time+1, :], axis=0)

        # Parameters for mean of each slice
        mean_params_slices = np.zeros((len(tumour_slices[0]), 3))
        mean_std_slices = np.zeros((len(tumour_slices[0]), 3))
        mean_Rsq_slices = np.zeros(len(tumour_slices[0]))
        for s in range(len(tumour_slices[0])):
            mean_params_slices[s, :], mean_std_slices[s, :], mean_Rsq_slices[s] = pmod.fit_brix_model(np.array([S_S0_ratio_mean_rm_bl[:, s]]), time_rm_bl, kwargs['tau'])

        S_S0_ratio_ROI_mean = np.mean(S_S0_ratio_mean, axis=1)
        start_time_list = [injection_time]
        #start_time_list = [7]
        start_time_name_list = ["rm_bl_"]
        for (start_time, time_name) in zip(start_time_list, start_time_name_list):
            S_S0_ratio_mean_rm_bl_temp = S_S0_ratio_ROI_mean[start_time:]
            S_S0_ratio_mean_rm_bl_temp[0] = np.mean(S_S0_ratio_ROI_mean[:(start_time+1)], axis=0)
            mean_param, mean_stds, mean_Rsq = pmod.fit_brix_model(np.array([S_S0_ratio_mean_rm_bl_temp]),
                                                          time[start_time:]-time[start_time], kwargs['tau'])
            mean_param_path = os.path.join(parameters_loc,
                                           "MeanROI_param_model_{}_tumour_{}_{}.npz".format(model, tumour_name, time_name))
            np.savez(mean_param_path, S_S0_mean=S_S0_ratio_mean, ROI_param=mean_param, ROI_std=mean_stds, ROI_Rsq=mean_Rsq,
                     mean_params_slices=mean_params_slices, mean_std_slices=mean_std_slices,
                     slices=tumour_slices, tau=tau)


def get_AUC_time_indx(time, AUC_time):
    AUC_closest = min(time, key=lambda x:abs(x-AUC_time))
    indx = np.where(time==AUC_closest)
    return indx[0][0]


def AUC_2D_calculation(CI, time, tumour_mask, tumour_name, AUC_time, dt, affine, AUC_dir):
    # READING CI (remove CI after chosen AUC time)
    AUC_time = int(AUC_time)
    AUC_time_idx = get_AUC_time_indx(time, AUC_time)
    CI_AUC_time = CI[:AUC_time_idx]

    # SLICES WITH TUMOUR
    tumour_pixels_in_slice = np.sum(tumour_mask, axis=(0, 1))
    tumour_slices = np.where(tumour_pixels_in_slice > 0)
    tumour_slices = tumour_slices[0]

    # CALCULATING THE AUC MAP
    nt, nx, ny, nz = CI_AUC_time.shape
    AUC_map = np.zeros((nx, ny, nz))
    CI_mean_slices = np.zeros((nt, len(tumour_slices))) # Need an array to store the mean CI for each slice
    for j, s_num in enumerate(tumour_slices):
        tumour_mask_slice = tumour_mask[:, :, s_num]
        CI_slice = CI_AUC_time[:, :, :, s_num]
        AUC_map[:, :, s_num], CI_mean_slices[:, j] = pmod.AUC_maps(CI_slice, tumour_mask_slice, dt)

    # STORING THE AUC MAP AS A NIFTI FILE
    AUC_map_path = os.path.join(AUC_dir, "AUC{}_tumour_{}_.nii.gz".format(AUC_time, tumour_name))
    nifti_img = nib.Nifti1Image(AUC_map, affine=affine)
    nib.save(nifti_img, AUC_map_path)

    # CALCULATE THE AUC FOR THE MEAN CI TAKEN OVER THE WHOLE TUMOUR
    CI_mean = np.mean(CI_mean_slices, axis=1)
    ROI_AUC = pmod.AUC_calc(np.array([CI_mean]), dt)

    return ROI_AUC

def TTHP_calculation(S, time, tumour_mask, tumour_name, TTHP_dir, affine):
    # SLICES WITH TUMOUR
    tumour_pixels_in_slice = np.sum(tumour_mask, axis=(0, 1))
    tumour_slices = np.where(tumour_pixels_in_slice > 0)
    tumour_slices = tumour_slices[0]

    # CALCULATING THE TTHP MAP
    nt, nx, ny, nz = S.shape
    TTHP_map = np.zeros((nx, ny, nz))
    S_mean_slices = np.zeros((nt, len(tumour_slices)))

    for j, s_num in enumerate(tumour_slices):
        print("Slice: {} {}".format(j, s_num))
        tumour_mask_slice = tumour_mask[:, :, s_num]
        S_slice = S[:, :, :, s_num]
        TTHP_map[:, :, s_num], S_mean_slices[:, j] = pmod.TTHP_map(S_slice, tumour_mask_slice, time)

    # STORING THE TTHP MAP AS A NIFTI FILE
    TTHP_map_path = os.path.join(TTHP_dir, "TTHP_tumour_{}_.nii.gz".format(tumour_name))
    nifti_img = nib.Nifti1Image(TTHP_map, affine=affine)
    nib.save(nifti_img, TTHP_map_path)

    # CALCULATING THE ROI TTHP
    # Averaging over the slices to get the mean signal over the WHOLE tumour
    S_mean = np.mean(S_mean_slices, axis=1)
    ROI_TTHP = pmod.TTHP_calc(np.array([S_mean]), time)
    ROI_TTHP = ROI_TTHP[0]

    # STORING THE ROI TTHP
    TTHP_ROI_path = os.path.join(TTHP_dir, "ROI_TTHP_tumour_{}_.npz".format(tumour_name))
    np.savez(TTHP_ROI_path, S=S_mean, TTHP=ROI_TTHP, time=time, tumour_name=tumour_name)


def remove_outliers(parameter1, parameter2):
    #Removing parameter that are outside the 1st and 99th percentile
    q1_parameter1, q3_parameter1 = np.percentile(sorted(parameter1), [25, 75])
    iqr_parameter1 = q3_parameter1 - q1_parameter1
    lower_bound = q1_parameter1 - (3 * iqr_parameter1)
    upper_bound = q3_parameter1 + (3 * iqr_parameter1)
    #lower_bound, upper_bound = np.percentile(sorted(parameter1), [1, 99])
    parameter1_wo_outliers_idx = np.argwhere(np.logical_or(parameter1 < lower_bound, parameter1 > upper_bound)).flatten()
    parameter1_wo_outliers_idx = parameter1_wo_outliers_idx.flatten()

    q1_parameter2, q3_parameter2 = np.percentile(sorted(parameter2), [25, 75])
    iqr_parameter2 = q3_parameter2 - q1_parameter2
    lower_bound = q1_parameter2 - (3 * iqr_parameter2)
    upper_bound = q3_parameter2 + (3 * iqr_parameter2)
    #lower_bound, upper_bound = np.percentile(sorted(parameter2), [1, 99])
    parameter2_wo_outliers_idx = np.argwhere(np.logical_or(parameter2 < lower_bound, parameter2 > upper_bound))
    parameter2_wo_outliers_idx = parameter2_wo_outliers_idx.flatten()

    parameter_wo_outliers_idx = np.append(parameter1_wo_outliers_idx, parameter2_wo_outliers_idx)
    parameter_wo_outliers_idx = np.unique(parameter_wo_outliers_idx)
    print("Number of outliers removed:", len(parameter_wo_outliers_idx))

    parameter1 = np.delete(parameter1, parameter_wo_outliers_idx)
    parameter2 = np.delete(parameter2, parameter_wo_outliers_idx)
    return parameter1, parameter2, len(parameter_wo_outliers_idx)

def remove_outliers_above_one(parameter1, parameter2):
    lower_bound = 0
    upper_bound = 1

    parameter1_wo_outliers_idx = np.argwhere(np.logical_or(parameter1 < lower_bound, parameter1 > upper_bound)).flatten()
    parameter1_wo_outliers_idx = parameter1_wo_outliers_idx.flatten()

    parameter2_wo_outliers_idx = np.argwhere(np.logical_or(parameter2 < lower_bound, parameter2 > upper_bound))
    parameter2_wo_outliers_idx = parameter2_wo_outliers_idx.flatten()

    parameter_wo_outliers_idx = np.append(parameter1_wo_outliers_idx, parameter2_wo_outliers_idx)
    parameter_wo_outliers_idx = np.unique(parameter_wo_outliers_idx)
    print("Number of outliers removed:", len(parameter_wo_outliers_idx))

    parameter1 = np.delete(parameter1, parameter_wo_outliers_idx)
    parameter2 = np.delete(parameter2, parameter_wo_outliers_idx)
    return parameter1, parameter2, len(parameter_wo_outliers_idx)

def remove_outliers_above_one_for_multiple_parameters(parameters):
    """
    Removed the outliers that have values below 0 and above 1 from all keys in the dictionary
    :param parameters: is a dictionary with pharmacokinetic parameters for each population AIF
    :return: new dictionary wihout outliers and which outliers were removed
    """
    lower_bound = 0
    upper_bound = 1

    for i, parameter1 in enumerate(parameters.values()):
        for j, parameter2 in enumerate(parameters.values()):
            if j > i:
                parameter1_wo_outliers_idx = np.argwhere(np.logical_or(parameter1 < lower_bound, parameter1 > upper_bound)).flatten()
                parameter1_wo_outliers_idx = parameter1_wo_outliers_idx.flatten()

                parameter2_wo_outliers_idx = np.argwhere(np.logical_or(parameter2 < lower_bound, parameter2 > upper_bound))
                parameter2_wo_outliers_idx = parameter2_wo_outliers_idx.flatten()

                parameter_wo_outliers_idx = np.append(parameter1_wo_outliers_idx, parameter2_wo_outliers_idx)

    parameter_wo_outliers_idx = np.unique(parameter_wo_outliers_idx)
    print("Number of outliers removed:", len(parameter_wo_outliers_idx))

    new_parameters = {}
    for popAIF, parameter in parameters.items():
        new_parameters[popAIF] = np.delete(parameter, parameter_wo_outliers_idx)

    return new_parameters, parameter_wo_outliers_idx

def pearsonr(matrix):
    r = np.corrcoef(matrix)
    rf = r[np.triu_indices(r.shape[0], 1)]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = p.T[np.tril_indices(p.shape[0], -1)]
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
    return r, p


def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Remove NaNs
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    df = df.dropna()
    y_true = df['y_true']
    y_pred = df['y_pred']

    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]

    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator

def ICC(calc_dcm, dir_path, model, parameter_type, popAIF_types, patients, values):
    exams = np.array(patients)
    for i in range(1, len(popAIF_types)):
        exams.append(patients)
    exams = np.array(exams).flatten()

    methods = np.array([[popAIF]*len(patients) for popAIF in popAIF_types])
    methods = methods.flatten()

    values = []
    for patient in patients:
        for popAIF_type in popAIF_types:
            parameter_path = os.path.join(calc_dcm, "PatientDatabase/{}_EMIN_{}_EMIN/Parameters/{}/MeanROI_param_"
                                                    "{}_p{}_model_{}_tumour_{}_.npz".format(patient, patient, model,
                                                                                            popAIF_type, patients, model, ))

