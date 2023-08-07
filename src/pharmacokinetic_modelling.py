################################################################################
######################## I M P O R T  P A C K A G E S ##########################
################################################################################

import time
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from scipy.integrate import cumtrapz, simps
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pydicom as pd

################################################################################
################################# M O D E L S ##################################
################################################################################

def ext_tofts_integral(t, Cp, Kt=0.1, ve=0.2, vp=0.1,
                       uniform_sampling=True):
    """ Extended Tofts Model, with time t in min.
        Works when t_dce = t_aif only and t is uniformly spaced.
    """
    nt = len(t)
    Ct = zeros(nt)
    for k in range(nt):
        if uniform_sampling:
            tmp = cumtrapz(exp(-Kt * (t[k] - t[:k + 1]) / ve) * Cp[:k + 1],
                           t[:k + 1], initial=0.0) + vp * Cp[:k + 1]
            Ct[k] = tmp[-1]
        else:
            Ct[k] = simps(exp(-Kt * (t[k] - t[:k + 1]) / ve) * Cp[:k + 1],
                          t[:k + 1]) + vp * Cp[:k + 1]
    return Ct * Kt


def tofts_integral(t, Cp, Kt=0.1, ve=0.2, uniform_sampling=True):
    ''' Standard Tofts Model, with time t in min.
        Current works only when AIF and DCE data are sampled on
        same grid.  '''
    nt = len(t)
    Ct = zeros(nt)
    for k in range(nt):
        if uniform_sampling:
            tmp = cumtrapz(exp(-(Kt / ve) * (t[k] - t[:k + 1])) * Cp[:k + 1],
                           t[:k + 1], initial=0.0)
            Ct[k] = tmp[-1]
            # Ct[k] = simps(exp(-(Kt/ve)*(t[k] - t[:k+1]))*Cp[:k+1],
            #              dx=t[1]-t[0])
        else:
            Ct[k] = simps(exp(-(Kt / ve) * (t[k] - t[:k + 1])) * Cp[:k + 1], x=t[:k + 1])
    return Kt * Ct


def Brix_model(time, tau, Kep, Kel, A):
    nt = len(time)
    S_S0_ratio = zeros(nt)

    for i, t in enumerate(time):
        if t < tau:
            S_S0_ratio[i] = 1 + A * ((np.exp(Kel * t) - 1) * np.exp(-Kel * t) / Kel - (np.exp(Kep * t) - 1) * np.exp(
                -Kep * t) / Kep) / (Kep - Kel)

        else:
            S_S0_ratio[i] = 1 + A * (
                        (np.exp(Kel * tau) - 1) * np.exp(-Kel * t) / Kel - (np.exp(Kep * tau) - 1) * np.exp(
                    -Kep * t) / Kep) / (Kep - Kel)

    # S_S0_ratio = 1 + A*Kep*((np.exp(-Kep*time)-np.exp(-Kel*time))/(Kel-Kep))

    return S_S0_ratio

def modified_Brix_model(time, Kep, Kel, A):
    nt = len(time)
    S_S0_ratio = zeros(nt)

    for i, t in enumerate(time):
        S_S0_ratio[i] = 1 + A * Kep * (np.exp(-Kep*t)-np.exp(-Kel*t)) / (Kel-Kep)

    return S_S0_ratio



def fit_tofts_model(Ct, Cp, t, idxs=None, extended=False,
                    plot_each_fit=False, ROI=False):
    ''' Solve tissue model for each voxel and return parameter maps.
        Ct: tissue concentration of CA, expected to be N x Ndyn
        t: time samples, assumed to be the same for Ct and Cp
        extended: if True, use Extended Tofts-Kety model.
        idxs: indices of ROI to fit
        '''
    print
    'fitting perfusion parameters'
    N, ndyn = Ct.shape
    Kt = zeros(N)
    ve = zeros(N)
    Kt_cov = zeros(N)
    ve_cov = zeros(N)
    R_sq = zeros(N)

    if idxs is None:
        idxs = range(N)

    # choose model and initialize fit parameters with reasonable values
    if extended:  # add vp if using Extended Tofts
        print('using Extended Tofts-Kety')
        vp = zeros(N)
        vp_cov = zeros(N)
        fit_func = lambda t, Kt, ve, vp: \
            ext_tofts_integral(t, Cp, Kt=Kt, ve=ve, vp=vp)
        coef0 = [1, 0.1, 0.1]
        popt_default = [-1, -1, -1]
        pcov_default = ones((3, 3))
    else:
        print('using Standard Tofts-Kety')
        fit_func = lambda t, Kt, ve: tofts_integral(t, Cp, Kt=Kt, ve=ve, uniform_sampling=False)
        coef0 = [1, 0.1] #[0.01, 0.1, 0.1]
        popt_default = [-1, -1]
        pcov_default = ones((2, 2))

    print('fitting %d voxels' % len(idxs))
    for k, idx in enumerate(idxs):
        try:
            popt, pcov = curve_fit(fit_func, t, Ct[idx, :], p0=coef0)
        except RuntimeError:
            popt = popt_default
            pcov = pcov_default
        Kt[idx] = popt[0]
        ve[idx] = popt[1]

        try:
            Kt_cov[idx] = pcov[0, 0]
            ve_cov[idx] = pcov[1, 1]
        except TypeError:
            None  # print idx, popt, pcov

        if extended:
            vp[idx] = popt[2]
            vp_cov[idx] = pcov[2, 2]

        if plot_each_fit:
            if k < 10:
                figure(1)
                clf()
                plot(t, Ct[idx, :], 'bo', alpha=0.6)
                plot(t, fit_func(t, *popt), 'm-')
                if extended:
                    title("Kt: {:.2f}, ve: {:.2f}, vp: {:.2f}".format(popt[0], popt[1], popt[2]))
                else:
                    title("Kt: {:.2f}, ve: {:.2f}".format(popt[0], popt[1]))
                pause(1)
                show()

        y = Ct[idx, :]
        if extended:
            y_fit = ext_tofts_integral(t, Cp, *popt)
        else:
            y_fit = tofts_integral(t, Cp, *popt)

        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        R_sq[idx] = 1 - (ss_res / ss_tot)

        #status_check(k, len(idxs), tstart=tstart)

    # bundle parameters for return
    params = [Kt, ve]
    stds = [sqrt(Kt_cov), sqrt(ve_cov)]
    if extended:
        params.append(vp)
        stds.append(sqrt(vp_cov))
    return (params, stds, R_sq)


def fit_brix_model(S_S0_ratio, time, tau, plot_each_fit=False, modified=False):
    N, ndyn = S_S0_ratio.shape
    Kep = zeros(N)
    Kel = zeros(N)
    A = zeros(N)
    Kep_cov = zeros(N)
    Kel_cov = zeros(N)
    A_cov = zeros(N)
    R_sq = zeros(N)
    idxs = range(N)

    if modified:
        fit_func = lambda t, Kep, Kel, A: modified_Brix_model(t, Kep=Kep, Kel=Kel, A=A)
        coef0 = [1, 0.01, 2.54]

    else:
        fit_func = lambda t, Kep, Kel, A: Brix_model(t, tau, Kep=Kep, Kel=Kel, A=A)
        #coef0 = [0.005, 0.004, 0.05]
        coef0 = [1, 0.01, 2.54]

    popt_default = [-1, -1, -1]
    pcov_default = ones((3, 3))
    lb = [-np.inf, 0.1, 0]
    ub = [np.inf, np.inf, 0.1]

    for k, idx in enumerate(idxs):
        try:
            popt, pcov = curve_fit(fit_func, time, S_S0_ratio[idx,:], coef0)


        except RuntimeError:
            popt = popt_default
            pcov = pcov_default

        Kep[idx] = popt[0]
        Kel[idx] = popt[1]
        A[idx] = popt[2]

        # Calculating R-squared statistics for the fitting
        y = S_S0_ratio[idx, :]
        if modified:
            y_fit = modified_Brix_model(time, *popt)
        else:
            y_fit = Brix_model(time, tau, *popt)
        ss_res = np.sum((y-y_fit)**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        R_sq[idx] = 1 - (ss_res/ss_tot)


        try:
            Kep_cov[idx] = pcov[0, 0]
            Kel_cov[idx] = pcov[1, 1]
            A_cov[idx] = pcov[2, 2]
        except TypeError:
            None  # print idx, popt, pcov
        if plot_each_fit:
            if k < 10:
                figure(1)
                clf()
                plot(time, y, 'bo', alpha=0.6)
                plot(time, Brix_model(time, tau, *popt), 'm-')
                title("Kep: {:.2f}, Kel: {:.4f}, A: {:.2f}".format(popt[0], popt[1], popt[2]))
                show()

    # bundle parameters for return
    params = [Kep, Kel, A]
    stds = [sqrt(Kep_cov), sqrt(Kel_cov), sqrt(A_cov)]

    return params, stds, R_sq



################################################################################
################# P A R A M E T E R  C A L C U L A T I O N S ###################
################################################################################

def parameter_maps(Ct, time, tumour_mask, model, **kwargs):
    """
    This function creates maps of the volume transfer constant (Kt)
    and extracellular volume ratio (ve) using the Tofts model and
    linear least square method for curve fitting.
    :param Ct: Tissue concentration of the region of interest (e.g. segmentation)
    :param Cp: AIF
    :param nx: number of rows of initial MRI data
    :param ny: number of columns of initial MRI data
    :param time: the time curve
    :param idxs_dce: the indices of the region of interest
    :param extended: if True the extended Tofts model is used instead of the normal one
    :return: maps (nx, ny) of Kt and ve
    """
    # Tumour mask
    tumour_seg_mask = tumour_mask.copy()
    tumour_seg_mask = tumour_seg_mask.flatten()
    tumour_seg_idx = np.where(tumour_seg_mask == True)

    # Readying AIF
    if model == 'TM' or model == 'ETM':
        Cp = kwargs['Cp']
        Cp = Cp.flatten()

    # Segmented concentration
    nt, nx, ny = Ct.shape
    Ct = np.transpose(Ct, (1, 2, 0))

    Ct = np.nan_to_num(Ct)
    Ct = np.reshape(Ct, (nx, ny, nt))
    Ct = np.reshape(Ct, (-1, nt))


    Ct_seg = Ct.copy()[tumour_seg_idx]

    # Calculate Kt and ve
    if model == "TM":
        params, stds, R_sq_temp = fit_tofts_model(Ct_seg, Cp, time, extended=False)
    elif model == "ETM":
        params, stds, R_sq_temp = fit_tofts_model(Ct_seg, Cp, time, extended=True)
    elif model == "Brix":
        params, stds, R_sq_temp = fit_brix_model(Ct_seg, time, kwargs['tau'])

    # Create maps of Kt and ve (Kep and Kel for Brix) with the right sizes
    Kt = np.zeros(nx * ny)
    Kt[tumour_seg_idx] = params[0]
    Kt = np.reshape(Kt, (nx, ny))

    ve = np.zeros(nx * ny)
    ve[tumour_seg_idx] = params[1]
    ve = np.reshape(ve, (nx, ny))

    # Create maps of stds of Kt (Kep for Brix) with the right sizes
    Kt_std = np.zeros(nx*ny)
    Kt_std[tumour_seg_idx] = stds[0]
    Kt_std = np.reshape(Kt_std, (nx, ny))

    # Create maps of stds of ve (Kel for Brix) with the right sizes
    ve_std = np.zeros(nx * ny)
    ve_std[tumour_seg_idx] = stds[1]
    ve_std = np.reshape(ve_std, (nx, ny))

    # Create maps of R^2 with the right sizes
    R_sq = np.zeros(nx*ny)
    R_sq[tumour_seg_idx] = R_sq_temp
    R_sq = np.reshape(R_sq, (nx, ny))

    if model == 'ETM' or model == 'Brix':
        # Create maps of vp (A for Brix) with the right sizes
        vp = np.zeros(nx * ny)
        vp[tumour_seg_idx] = params[2]
        vp = np.reshape(vp, (nx, ny))

        # Create maps of stds of vp (A for Brix) with the right sizes
        vp_std = np.zeros(nx * ny)
        vp_std[tumour_seg_idx] = stds[2]
        vp_std = np.reshape(vp_std, (nx, ny))


        return Kt, ve, vp, Kt_std, ve_std, vp_std, R_sq, tumour_seg_idx

    return Kt, ve, Kt_std, ve_std, R_sq, tumour_seg_idx


def AUC_calc(CI, dt):
    N, ndyn = CI.shape
    AUC = np.zeros(N)

    for indx in range(N):
        try:
            AUC_val = simps(CI[indx, :], dx=dt)

        except RuntimeError:
            AUC_val = -1

        AUC[indx] = AUC_val

    return AUC


def AUC_maps(CI, tumour_mask, dt):
    # Tumour mask
    tumour_mask = tumour_mask.copy()
    tumour_mask = tumour_mask.flatten()
    tumour_indx = np.where(tumour_mask == True)

    # Reshape CI and retrieve the CI in the tumour segmentation
    nt, nx, ny = CI.shape
    CI = np.transpose(CI, (1, 2, 0))
    CI = np.reshape(CI, (nx, ny, nt))
    CI = np.reshape(CI, (-1, nt))
    CI_seg = CI.copy()[tumour_indx]

    # Calculate the area under the curve for each voxel in the tumour
    AUC = AUC_calc(CI_seg, dt)
    AUC_map = np.zeros(nx * ny)
    AUC_map[tumour_indx] = AUC
    AUC_map = np.reshape(AUC_map, (nx, ny))

    # Calculate the mean CI taken over the tumour slice which will be used to calculate the area under the curve of the
    # mean CI over the whole tumour
    CI_mean = np.mean(CI_seg, axis=0)

    return AUC_map, CI_mean


def find_TTHP(S, time, indx):
    """
    :param list: The DCE-MRI ignal
    :param value: Half of the peak of the signal
    :return: time-to-half-peak
    """

    peak_S = np.max(S)
    half_peak_S = peak_S / 2

    number_of_noisy_voxels = 0
    if peak_S-S[0] > 50:
        # Find the indx of the peak
        peak_S_indx = np.argmax(S)

        # Find the point that is closest to the half-peak value
        diff = np.abs(S[:peak_S_indx] - half_peak_S)  # Remove the values after peak to make sure the TTHP is not after peak
        diff_sorted_indx = np.argsort(diff)
        nearest_point1_indx = diff_sorted_indx[0]
        nearest_point1 = S[nearest_point1_indx]

        similar_point = True
        i = 1
        while similar_point:
            if nearest_point1 < half_peak_S:
                nearest_point2_indx = nearest_point1_indx + i
            else:
                nearest_point2_indx = nearest_point1_indx - i

            nearest_point2 = S[nearest_point2_indx]
            if nearest_point1 == nearest_point2:
                i += 1
            else:
                similar_point = False

        nearest_points_y = np.array([nearest_point1, nearest_point2])
        nearest_points_x = time[[nearest_point1_indx, nearest_point2_indx]]

        # Find the slope and intercept of the line going through the two points
        res = linregress(nearest_points_x, nearest_points_y)
        slope = res.slope
        intercept = res.intercept

        # Find the time that corresponds to the half-peak value
        TTHP = (half_peak_S - intercept) / slope

    else:
        TTHP = -1

    return TTHP


def TTHP_calc(S, time):
    N, ndyn = S.shape
    TTHP = np.zeros(N)

    number_of_noisy_voxels = 0
    for indx in range(N):
        try:
            TTHP_val = find_TTHP(S[indx, :], time, indx)

        except RuntimeError:
            TTHP_val = -1

        TTHP[indx] = TTHP_val

        if TTHP_val == -1:
            number_of_noisy_voxels += 1

    print("{} noisy voxels out of {}".format(number_of_noisy_voxels, N))

    return TTHP

def TTHP_map(S, tumour_mask, time):
    # Tumour mask
    tumour_mask = tumour_mask.copy()
    tumour_mask = tumour_mask.flatten()
    tumour_indx = np.where(tumour_mask == True)

    # Reshape and retrieve the signal in the tumour segmentation
    nt, nx, ny = S.shape
    S = np.transpose(S, (1, 2, 0))
    S = np.reshape(S, (nx, ny, nt))
    S = np.reshape(S, (-1, nt))
    S_seg = S.copy()[tumour_indx]

    # Calculate the time-to-half-peak for each voxel in the tumour
    TTHP = TTHP_calc(S_seg, time)
    TTHP_map = np.zeros(nx*ny)
    TTHP_map[tumour_indx] = TTHP
    TTHP_map = np.reshape(TTHP_map, (nx, ny))

    # Calculate the mean signal over the tumour slice which will be used to calculate the TTHP for the mean signal over
    # the whole tumour
    S_mean = np.mean(S_seg, axis=0)

    return TTHP_map, S_mean



