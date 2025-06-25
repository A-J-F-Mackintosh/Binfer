import nlopt
from functools import partial
import math
import numpy as np
import moments
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("default", category=DeprecationWarning)
np.set_printoptions(linewidth=np.inf)

################################################ SFS functions ################################################

# from dadi (Gutenkunst et al. 2009; Blischak et al. 2020)
def three_epoch_inbreeding(params, ns, pts):
    """
    params = (nuB,nuF,TB,TF,F)
    ns = (n1,)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations) 
    TF: Time since bottleneck recovery (in units of 2*Na generations) 
    F: Inbreeding coefficent

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration (50-500 seems to work well)
    """
    nuB,nuF,TB,TF,F = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    if nuB and TB:
    	phi = Integration.one_pop(phi, xx, TB, nuB)
    if nuF and TF:
    	phi = Integration.one_pop(phi, xx, TF, nuF)

    fs = Spectrum.from_phi_inbreeding(phi, ns, (xx,), (F,), (2,))
    return fs

def four_epoch_inbreeding(params, ns, pts):
    """
    params = (nu2,nu1,nu0,T2,T1,0)
    ns = (n1,)

    nu2: Ratio of Ne in epoch2 to Ne in epoch3
    nu1: Ratio of Ne in epoch1 to Ne in epoch3
    nu0: Ratio of Ne in epoch0 to Ne in epoch3
    T2: Length of epoch2 (in units of 2*Na generations) 
    T1: Length of epoch2 (in units of 2*Na generations) 
    T0: Length of epoch2 (in units of 2*Na generations)  
    F: Inbreeding coefficent

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration (50-500 seems to work well)
    """
    nu2,nu1,nu0,T2,T1,T0,F = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    if nu2 and T2:
    	phi = Integration.one_pop(phi, xx, T2, nu2)
    if nu1 and T1:
    	phi = Integration.one_pop(phi, xx, T1, nu1)
    if nu0 and T0:
    	phi = Integration.one_pop(phi, xx, T0, nu0)

    fs = Spectrum.from_phi_inbreeding(phi, ns, (xx,), (F,), (2,))
    return fs

def moments_six_epoch(params, ns, pop_ids=None):
	"""
	Six epoch model of constant pop size
	"""
	nu0, nu1, nu2, nu3, nu4, T0, T1, T2, T3, T4 = params
	sts = moments.LinearSystem_1D.steady_state_1D(ns[0])
	fs = moments.Spectrum(sts, pop_ids=pop_ids)
	fs.integrate([nu0], T0, 0.02)
	fs.integrate([nu1], T1, 0.02)
	fs.integrate([nu2], T2, 0.02)
	fs.integrate([nu3], T3, 0.02)
	fs.integrate([nu4], T4, 0.02)
	return fs

def moments_four_epoch(params, ns, pop_ids=None):
	"""
	Four epoch model of constant pop size
	"""
	nu0, nu1, nu2, T0, T1, T2 = params
	sts = moments.LinearSystem_1D.steady_state_1D(ns[0])
	fs = moments.Spectrum(sts, pop_ids=pop_ids)
	fs.integrate([nu0], T0, 0.02)
	fs.integrate([nu1], T1, 0.02)
	fs.integrate([nu2], T2, 0.02)
	return fs


def get_1N_SFS_selfing(N0, mu, alpha, genomes):
	F = alpha / (2 - alpha)
	N0 = N0 / (1 + F)
	theta = mu * N0 * 4
	sfs = three_epoch_inbreeding((None, None, None, None, F), (genomes,), 100) * theta
	sfs[0] = 1 - sum(sfs[1:genomes])
	sfs = sfs[0:genomes]
	return sfs


# can also be used as 1N + ND
def get_2N_SFS_selfing(N0, N1, step, mu, alpha, ploidy, genomes):
	F = alpha / (2 - alpha)
	N0 = N0 / (1 + F)
	N1 = N1 / (1 + F)
	theta = mu * N1 * 4
	if ploidy == 2:
		sfs = three_epoch_inbreeding((N0/N1, None, step / (2*N1), None, F), (genomes,), 100) * theta
	elif ploidy == 1:
		sfs = moments.Demographics1D.two_epoch((N0 / N1, step / (N1*2)), [genomes], pop_ids=None) * theta
	sfs[0] = 1 - sum(sfs[1:genomes])
	sfs = sfs[0:genomes]
	return sfs


def get_2N_SFS_selfing_ND(ND_step_1, ND_step_2, step, N0, N1, B0, B1, alpha, mu, genomes):
	F = alpha / (2 - alpha)
	N0 = N0 / (1 + F)
	N1 = N1 / (1 + F)
	theta = mu * N1 * 4 * B1
	if ND_step_2 < step:
		sfs = four_epoch_inbreeding(((N0*B0)/(N1*B1), (((N0*B0)+N0)/2)/(N1*B1), N0/(N1*B1), 
			(step - ND_step_2) / (B1*N1*2), (ND_step_2 - ND_step_1) / (B1*N1*2), ND_step_1 / (B1*N1*2), F), 
		(genomes,), 100) * theta
	elif ND_step_1 < step < ND_step_2:
		sfs = four_epoch_inbreeding(((((N1*B1)+N1)/2)/(N1*B1), (((N0*B0)+N0)/2)/(N1*B1), N0/(N1*B1), 
			(ND_step_2 - step) / (B1*N1*2), (step - ND_step_1) / (B1*N1*2), ND_step_1 / (B1*N1*2), F), 
		(genomes,), 100) * theta
	else:
		return np.array([-1] * (genomes-1))
	sfs[0] = 1 - sum(sfs[1:genomes])
	sfs = sfs[0:genomes]
	return sfs


def get_1N_SFS(N0, mu, genomes):
	theta = mu * N0 * 4
	sfs = np.array([1 / k for k in range(1, genomes)])
	#r_sfs = np.array(list(reversed(sfs)))
	#sfs = np.append(np.append(np.array([0]), np.add(sfs, r_sfs)), np.array([0])) * theta
	sfs = np.append(np.array([0]), sfs) * theta
	#sfs[10] = sfs[10] / 2
	sfs[0] = 1 - np.sum(sfs[1:genomes])
	#sfs[11:22] = 0
	return sfs


def get_2N_SFS(N0, N1, step, mu, genomes):
	theta = mu * N1 * 4
	sfs = moments.Demographics1D.two_epoch((N0 / N1, step / (N1*2)), [genomes], pop_ids=None) * theta
	sfs[0] = 1 - sum(sfs[1:genomes])
	sfs = sfs[0:genomes]
	return sfs

# no theta, uses just the shape of the SFS to estimate two-epoch params
def get_2N_SFS_ratio(N_ratio, step_ratio, mu, genomes):
	sfs = moments.Demographics1D.two_epoch((N_ratio, step_ratio), [genomes], pop_ids=None)
	sfs = sfs[1:genomes]
	sfs = np.array([i / np.sum(sfs) for i in sfs]) # normalised SFS
	return sfs

def get_4N_SFS_ratio(N_ratio_0, N_ratio_1, N_ratio_2, step_ratio_0, step_ratio_1, step_ratio_2, mu, genomes):
	sfs = moments_four_epoch((N_ratio_2, N_ratio_1, N_ratio_0, 
		step_ratio_2 - step_ratio_1, step_ratio_1 - step_ratio_0, step_ratio_0), [genomes], pop_ids=None)
	sfs = sfs[1:genomes]
	sfs = np.array([i / np.sum(sfs) for i in sfs]) # normalised SFS
	return sfs


def get_2N_SFS_ND(ND_step, step, N0, N1, B0, B1, mu, genomes):
	theta = mu * N1 * B1 * 4
	if ND_step < step:
		sfs = moments.Demographics1D.three_epoch(((N0*B0) / (N1*B1), N0 / (N1*B1), 
			(step - ND_step) / (B1*N1*2), ND_step / (B1*N1*2)), [genomes], pop_ids=None) * theta
	else:
		sfs = moments.Demographics1D.three_epoch(((N1) / (N1*B1), N0 / (N1*B1), 
			(ND_step - step) / (B1*N1*2), step / (B1*N1*2)), [genomes], pop_ids=None) * theta

	sfs[0] = 1 - sum(sfs[1:genomes])
	sfs = sfs[0:genomes]
	return sfs

def get_4N_SFS_ND(ND_step_1, ND_step_2, step0, step1, step2, 
	N0, N1, N2, N3, B0, B1, B2, B3, mu, genomes):
	theta = mu * N3 * B3 * 4

	if ND_step_2 < step0:
		sfs = moments_six_epoch((
			(N2*B2) / (N3*B3), 
			(N1*B1) / (N3*B3), 
			(N0*B0) / (N3*B3), 
			(N0*B0 + ((N0 - N0*B0)/2)) / (N3*B3), 
			N0 / (N3*B3), 
			(step2 - step1) / (B3*N3*2), 
			(step1 - step0) / (B3*N3*2), 
			(step0 - ND_step_2) / (B3*N3*2), 
			(ND_step_2 - ND_step_1) / (B3*N3*2), 
			ND_step_1 / (B3*N3*2)
			), 
		[genomes], pop_ids=None) * theta

	elif ND_step_1 < step0 < ND_step_2 < step1:
		sfs = moments_six_epoch((
			(N2*B2) / (N3*B3), 
			(N1*B1) / (N3*B3), 
			(N1*B1 + ((N1 - N1*B1)/2)) / (N3*B3), 
			(N0*B0 + ((N0 - N0*B0)/2)) / (N3*B3), 
			N0 / (N3*B3), 
			(step2 - step1) / (B3*N3*2), 
			(step1 - ND_step_2) / (B3*N3*2), 
			(ND_step_2 - step0) / (B3*N3*2), 
			(step0 - ND_step_1) / (B3*N3*2), 
			ND_step_1 / (B3*N3*2)
			), 
		[genomes], pop_ids=None) * theta

	elif step0 < ND_step_1 < ND_step_2 < step1:
		sfs = moments_six_epoch((
			(N2*B2) / (N3*B3), 
			(N1*B1) / (N3*B3), 
			(N1*B1 + ((N1 - N1*B1)/2)) / (N3*B3), 
			N1 / (N3*B3), 
			N0 / (N3*B3), 
			(step2 - step1) / (B3*N3*2), 
			(step1 - ND_step_2) / (B3*N3*2), 
			(ND_step_2 - ND_step_1) / (B3*N3*2), 
			(ND_step_1 - step0) / (B3*N3*2), 
			step0 / (B3*N3*2)
			), 
		[genomes], pop_ids=None) * theta

	elif step0 < ND_step_1 < step1 < ND_step_2 < step2:
		sfs = moments_six_epoch((
			(N2*B2) / (N3*B3), 
			(N2*B2 + ((N2 - N2*B2)/2)) / (N3*B3), 
			(N1*B1 + ((N1 - N1*B1)/2)) / (N3*B3), 
			N1 / (N3*B3), 
			N0 / (N3*B3), 
			(step2 - ND_step_2) / (B3*N3*2), 
			(ND_step_2 - step1) / (B3*N3*2), 
			(step1 - ND_step_1) / (B3*N3*2), 
			(ND_step_1 - step0) / (B3*N3*2), 
			step0 / (B3*N3*2)
			), 
		[genomes], pop_ids=None) * theta

	elif ND_step_1 < step0 < step1 < ND_step_2 < step2:
		sfs = moments_six_epoch((
			(N2*B2) / (N3*B3), 
			(N2*B2 + ((N2 - N2*B2)/2)) / (N3*B3), 
			(N1*B1 + ((N1 - N1*B1)/2)) / (N3*B3), 
			(N0*B0 + ((N0 - N0*B0)/2)) / (N3*B3), 
			N0 / (N3*B3), 
			(step2 - ND_step_2) / (B3*N3*2), 
			(ND_step_2 - step1) / (B3*N3*2), 
			(step1 - step0) / (B3*N3*2), 
			(step0 - ND_step_1) / (B3*N3*2), 
			ND_step_1 / (B3*N3*2)
			), 
		[genomes], pop_ids=None) * theta

	elif step0 < step1 < ND_step_1 < ND_step_2 < step2:
		sfs = moments_six_epoch((
			(N2*B2) / (N3*B3), 
			(N2*B2 + ((N2 - N2*B2)/2)) / (N3*B3), 
			N2 / (N3*B3), 
			N1 / (N3*B3), 
			N0 / (N3*B3), 
			(step2 - ND_step_2) / (B3*N3*2), 
			(ND_step_2 - ND_step_1) / (B3*N3*2), 
			(ND_step_1 - step1) / (B3*N3*2), 
			(step1 - step0) / (B3*N3*2), 
			step0 / (B3*N3*2)
			), 
		[genomes], pop_ids=None) * theta

	else:
		return np.array([-1] * (genomes-1))

	sfs[0] = 1 - sum(sfs[1:genomes])
	sfs = sfs[0:genomes]

	#print("##", sfs)

	return sfs