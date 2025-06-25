import sys
import time
import random
import math
import numpy as np
import moments
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("default", category=DeprecationWarning)
np.set_printoptions(linewidth=np.inf)

import contextlib
import multiprocessing

import SFS as BinfS
import B_map as BinfB


########################################### multiprocessing functions ###########################################

@contextlib.contextmanager
def poolcontext(*args, **kwargs):
	pool = multiprocessing.Pool(*args, **kwargs)
	yield pool
	pool.terminate()

def chunks(lst, n):
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


################################################## likelihood functions ################################################

def SFS_to_pi(SFS, genomes):
	n = genomes
	het = 0
	for i in range(0, len(SFS)):
		if SFS[i] > 0:
			het += SFS[i] * i*(n - i) / (n * (n - 1) * 0.5)
	return np.array([het, np.sum(SFS) - het])

def polarisation_error(sfs, eps, init):
	new_sfs = np.zeros(len(sfs))
	new_sfs[0] = sfs[0]
	n = len(new_sfs) - 1
	for i in range(init, int(math.ceil((len(sfs)-init) / 2.0))+init):
		low_i = sfs[i] - sfs[i]*eps + sfs[n-i+init]*eps
		high_i = sfs[n-i+init] - sfs[n-i+init]*eps + sfs[i]*eps
		new_sfs[i] = low_i
		new_sfs[n-i+init] = high_i
	return new_sfs


def optimisation_theta(params, grad, observed_SFS, other_params, genomes, epochs, alpha, mask_hf):
	
	theta = params[0]
	theta = 10**theta
	other_params = [10**p for p in other_params]

	if alpha:

		if epochs == 2:

			if len(other_params) == 3:
				N_ratio, step_ratio, eps = other_params
			elif len(other_params) == 4:
				N_ratio, step_ratio, eps, alpha = other_params

			sfs = BinfS.three_epoch_inbreeding((N_ratio, None, step_ratio, None, alpha / (2 - alpha)), (genomes,), 100) * theta
			sfs[0] = 1 - sum(sfs[1:genomes])
			sfs = sfs[0:genomes]
			sfs = polarisation_error(sfs, eps, 1)

		elif epochs == 3:

			N0_ratio, N1_ratio, step0_ratio, step1_ratio, eps = other_params

			sfs = BinfS.three_epoch_inbreeding((N1_ratio, N0_ratio, step1_ratio - step0_ratio, step0_ratio, alpha / (2 - alpha)), (genomes,), 100) * theta
			sfs[0] = 1 - sum(sfs[1:genomes])
			sfs = sfs[0:genomes]
			sfs = polarisation_error(sfs, eps, 1)

	else:

		if epochs == 2:
			N_ratio, step_ratio, eps = other_params
			sfs = moments.Demographics1D.two_epoch((N_ratio, step_ratio), [genomes], pop_ids=None) * theta
			sfs[0] = 1 - sum(sfs[1:genomes])
			sfs = sfs[0:genomes]
			sfs = polarisation_error(sfs, eps, 1)

		elif epochs == 3:
			N_ratio_0, N_ratio_1, step_ratio_0, step_ratio_1, eps = other_params
			sfs = BinfS.moments_four_epoch((1.0, N_ratio_1, N_ratio_0, 
			step_ratio_1 * 1.1 - step_ratio_1, step_ratio_1 - step_ratio_0, step_ratio_0), [genomes], pop_ids=None) * theta
			sfs[0] = 1 - sum(sfs[1:genomes])
			sfs = sfs[0:genomes]
			sfs = polarisation_error(sfs, eps, 1)

		elif epochs == 4:
			N_ratio_0, N_ratio_1, N_ratio_2, step_ratio_0, step_ratio_1, step_ratio_2 = other_params
			sfs = BinfS.moments_four_epoch((N_ratio_2, N_ratio_1, N_ratio_0, 
			step_ratio_2 - step_ratio_1, step_ratio_1 - step_ratio_0, step_ratio_0), [genomes], pop_ids=None) * theta
			sfs[0] = 1 - sum(sfs[1:genomes])
			sfs = sfs[0:genomes]
			sfs = polarisation_error(sfs, eps, 1)

	if sfs[0] < 0:
		return float("-inf")

	observed_SFS_copy = np.copy(observed_SFS)

	# mask reference bias counts
	if mask_hf:
		sfs[0] += sfs[-1]
		sfs[0] += sfs[-2]
		sfs = sfs[0:genomes-2]

		observed_SFS_copy[0] += observed_SFS_copy[-1]
		observed_SFS_copy[0] += observed_SFS_copy[-2]
		observed_SFS_copy = observed_SFS_copy[0:genomes-2]

	log_expected_SFS = np.log(sfs) # this will create a warning if there is a zero in there
	temp_likelihoods = np.multiply(log_expected_SFS, observed_SFS_copy) # this will lead to -inf*0 -> nan, or -inf*[>0] -> -inf, nan is fine!
	loglikelihood = np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)) # change nan to zero

	return loglikelihood


def optimisation_1N(params, grad, mu, observed_SFS, verbose, mode, genomes):

	if len(params) == 1:
		Ne0 = params[0]
		alpha = 0
	else:
		Ne0, alpha = params

	if mode == "SFS":
		if alpha == 0:
			try:
				expected_SFS = BinfS.get_1N_SFS(Ne0, mu, genomes)
			except:
				return float("-inf")
		else:
			try:
				expected_SFS = BinfS.get_1N_SFS_selfing(Ne0, mu, alpha, genomes)
			except:
				return float("-inf")

	elif mode == "pi":
		try:
			expected_SFS = np.array([Ne0 * 4 * mu, 1 - (Ne0 * 4 * mu)])
		except:
			return float("-inf")

	if (expected_SFS < 0).sum() > 0:
		return float("-inf")

	else:
		log_expected_SFS = np.log(expected_SFS) # this will create a warning if there is a zero in there
		temp_likelihoods = np.multiply(log_expected_SFS, observed_SFS) # this will lead to -inf*0 -> nan, or -inf*[>0] -> -inf, nan is fine!
		loglikelihood = np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)) # change nan to zero

	return loglikelihood


def optimisation_multi_epoch(params, grad, mu, observed_SFS, verbose, genomes, epochs, alpha, mask_hf, eps):

	# normalise
	if mask_hf:
		# mask ref bias counts
		n_observed_SFS = np.array([i / np.sum(observed_SFS[1:genomes-2]) for i in observed_SFS[1:genomes-2]])
	else:
		n_observed_SFS = np.array([i / np.sum(observed_SFS[1:genomes]) for i in observed_SFS[1:genomes]])

	params = [10**p for p in params]

	if alpha:

		if epochs == 2:

			if eps:
				if len(params) == 2:
					N_ratio, step_ratio = params
				elif len(params) == 3:
					N_ratio, step_ratio, alpha = params
			else:
				if len(params) == 3:
					N_ratio, step_ratio, eps = params
				elif len(params) == 4:
					N_ratio, step_ratio, eps, alpha = params

			F = alpha / (2 - alpha)
			try:
				sfs = BinfS.three_epoch_inbreeding((N_ratio, None, step_ratio, None, F), (genomes,), 100)
				sfs = sfs[1:genomes]
				sfs = polarisation_error(sfs, eps, 0)
				expected_SFS = np.array([i / np.sum(sfs) for i in sfs])
			except:
				return float("-inf")

		elif epochs == 3:

			if eps:
				N_ratio_0, N_ratio_1, step_ratio_0, step_ratio_1 = params
			else:
				N_ratio_0, N_ratio_1, step_ratio_0, step_ratio_1, eps = params

			F = alpha / (2 - alpha)
			try:
				sfs = BinfS.three_epoch_inbreeding((N_ratio_1, N_ratio_0, step_ratio_1 - step_ratio_0, step_ratio_0, F), (genomes,), 100)
				sfs = sfs[1:genomes]
				sfs = polarisation_error(sfs, eps, 0)
				expected_SFS = np.array([i / np.sum(sfs) for i in sfs])
			except:
				return float("-inf")

	else:

		if epochs == 2:

			if eps:
				N_ratio, step_ratio = params
			else:
				N_ratio, step_ratio, eps = params

			try:
				expected_SFS = BinfS.get_2N_SFS_ratio(N_ratio, step_ratio, mu, genomes)
				expected_SFS = polarisation_error(expected_SFS, eps, 0)
			except:
				return float("-inf")

		elif epochs == 3:

			if eps:
				N_ratio_0, N_ratio_1, step_ratio_0, step_ratio_1 = params
			else:
				N_ratio_0, N_ratio_1, step_ratio_0, step_ratio_1, eps = params

			if not step_ratio_0 < step_ratio_1:
				return float("-inf")
			try:
				expected_SFS = BinfS.get_4N_SFS_ratio(N_ratio_0, N_ratio_1, 1.0, 
					step_ratio_0, step_ratio_1, 1.1 * step_ratio_1, mu, genomes)
				expected_SFS = polarisation_error(expected_SFS, eps, 0)
			except:
				return float("-inf")

		elif epochs == 4:

			if eps:
				N_ratio_0, N_ratio_1, N_ratio_2, step_ratio_0, step_ratio_1, step_ratio_2 = params
			else:
				N_ratio_0, N_ratio_1, N_ratio_2, step_ratio_0, step_ratio_1, step_ratio_2, eps = params

			if not step_ratio_0 < step_ratio_1 < step_ratio_2:
				return float("-inf")
			try:
				expected_SFS = BinfS.get_4N_SFS_ratio(N_ratio_0, N_ratio_1, N_ratio_2, 
					step_ratio_0, step_ratio_1, step_ratio_2, mu, genomes)
				expected_SFS = polarisation_error(expected_SFS, eps, 0)
			except:
				return float("-inf")

	if (expected_SFS < 0).sum() > 0:
			return float("-inf")

	else:
		#print(params)
		#print(n_observed_SFS)

		# mask ref bias counts
		if mask_hf:
			expected_SFS = np.array([i / np.sum(expected_SFS[0:genomes-3]) for i in expected_SFS[0:genomes-3]])
		
		#print(expected_SFS, "\n")

		log_expected_SFS = np.log(expected_SFS) # this will create a warning if there is a zero in there
		temp_likelihoods = np.multiply(log_expected_SFS, n_observed_SFS) # this will lead to -inf*0 -> nan, or -inf*[>0] -> -inf, nan is fine!
		loglikelihood = np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)) # change nan to zero

	return loglikelihood


def optimisation_1N_classicBGS(params, grad, mu, observed_SFS_windows, verbose, 
	r_distance_array, exon_array, r_array, mode, ploidy, genomes, observed_Ne, round_B, alpha, window_size):

	if len(params) == 3:
		Ne0 = params[0]
		u = mu
		s_mean = params[1]
		s_shape = params[2]
		h = 0.5

	elif len(params) == 4:
		Ne0 = params[0]
		u = params[1]
		s_mean = params[2]
		s_shape = params[3]
		h = 0.5
		
	elif len(params) == 5:
		Ne0 = params[0]
		u = params[1]
		s_mean = params[2]
		s_shape = params[3]
		h = params[5]

	loglikelihood = 0

	if alpha:
		F = alpha / (2 - alpha)
	else:
		F = 0

	min_s = 3 / (observed_Ne * 2) / (h*(1 - F) + F)

	B_array = BinfB.get_B_map(s_shape, s_mean, u, alpha, h, min_s, r_distance_array, r_array, exon_array, window_size)

	if B_array is None:
		print(params, "B-map calculation failed", flush=True)
		return float("-inf")

	if alpha:

		SFS_lookup_BGS = {}

		for idx, window_SFS in enumerate(observed_SFS_windows):

			if np.sum(window_SFS) == 0:
				pass

			else:
				if round_B:
					SFS_key = tuple([round(B_array[idx], 2)])
					if SFS_key not in SFS_lookup_BGS.keys():
						try:
							SFS_lookup_BGS[SFS_key] = BinfS.get_1N_SFS_selfing(Ne0*round(B_array[idx], 2), mu, alpha, genomes)
							
							if (SFS_lookup_BGS[SFS_key] <= 0).sum() > 0 or (SFS_lookup_BGS[SFS_key] >= 1).sum() > 0 or np.isnan(np.sum(SFS_lookup_BGS[SFS_key])):
								return float("-inf")

							expected_SFS = SFS_lookup_BGS[SFS_key]

						except:
							return float("-inf")
					else:
						expected_SFS = SFS_lookup_BGS[SFS_key]

				else:
					try:
						expected_SFS = BinfS.get_1N_SFS_selfing(Ne0*B_array[idx], mu, alpha, genomes)
						if (expected_SFS <= 0).sum() > 0 or (expected_SFS >= 1).sum() > 0 or np.isnan(np.sum(expected_SFS)):
							return float("-inf")
					except:
						return float("-inf")

				log_expected_SFS = np.log(expected_SFS) # this will create a warning if there is a zero in there
				temp_likelihoods = np.multiply(log_expected_SFS, window_SFS) # this will lead to -inf*0 -> nan, or -inf*[>0] -> -inf, nan is fine!
				loglikelihood += np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)) # change nan to zero

				# if loglikelihood >= 0:
				# 	print(expected_SFS)
				# 	print(log_expected_SFS)
				# 	print(temp_likelihoods)
				# 	print(np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)))
				# 	sys.exit()

	else:

		E_pi = B_array * (4 * Ne0 * mu)
		P_mismatch = np.array([np.array([e_pi / (1 + e_pi), 1 - (e_pi / (1 + e_pi))]) for e_pi in E_pi])

		for idx, window_SFS in enumerate(observed_SFS_windows):

			if np.sum(window_SFS) == 0:
				pass

			else:

				# probably better to do this once, outside of optimisation
				focal_window_mm = SFS_to_pi(window_SFS, genomes)

				log_expected_mm = np.log(P_mismatch[idx]) # this will create a warning if there is a zero in there
				temp_likelihoods = np.multiply(log_expected_mm, focal_window_mm) # this will lead to -inf*0 -> nan, or -inf*[>0] -> -inf, nan is fine!
				loglikelihood += np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)) # change nan to zero

	print(params, loglikelihood, flush=True)

	return loglikelihood


def optimisation_3N_classicBGS(params, grad, mu, observed_SFS_windows, verbose, 
	r_distance_array, exon_array, r_array, mode, ploidy, genomes, observed_Ne, maximize, precision, alpha, 
	mask_hf, eps, window_size):

	if alpha:
		sys.exit("not implemented")

	if len(params) == 8:
		Ne0 = params[0]
		Ne1 = params[1]
		Ne2 = params[2]
		step0 = params[3]
		step1 = params[4]
		u = params[5]
		s_mean = params[6]
		s_shape = params[7]
		h = 0.5

	loglikelihood = 0

	if alpha:
		F = alpha / (2 - alpha)
	else:
		F = 0

	if step0 > step1:
		if maximize:
			return float("-inf")
		else:
			return float("inf")

	overall_Nmax = overall_Ne_recursion(0, 0, 0, [Ne0, step0, Ne1, step1, Ne2, float("inf")], 0)

	approx_reduction = observed_Ne / overall_Nmax

	min_s_0 = 3 / (approx_reduction * 2 * Ne0) / (h*(1 - F) + F)
	min_s_1 = 3 / (approx_reduction * 2 * Ne1) / (h*(1 - F) + F)
	min_s_2 = 3 / (approx_reduction * 2 * Ne2) / (h*(1 - F) + F)

	B_array_0 = BinfB.get_B_map(s_shape, s_mean, u, alpha, h, min_s_0, r_distance_array, r_array, exon_array, window_size)
	B_array_1 = BinfB.get_B_map(s_shape, s_mean, u, alpha, h, min_s_1, r_distance_array, r_array, exon_array, window_size)
	B_array_2 = BinfB.get_B_map(s_shape, s_mean, u, alpha, h, min_s_2, r_distance_array, r_array, exon_array, window_size)

	if B_array_0 is None or B_array_1 is None or B_array_2 is None:
		if maximize:
			return float("-inf")
		else:
			return float("inf")

	if precision == 0:
		max_rs_iters = 1
		max_thresh_diff = 0.1
	elif precision == 1:
		max_rs_iters = 5
		max_thresh_diff = 0.01
	else:
		max_rs_iters = 10
		max_thresh_diff = 0.001

	prev_thresh = min_s_0
	thresh_diff = float("infinity")
	rs_iters = 0

	while thresh_diff > max_thresh_diff and rs_iters < max_rs_iters:

		min_s_0 = 3 / (np.mean(B_array_0) * 2 * Ne0 / (1 + F)) / (h*(1 - F) + F)
		B_array_0 = BinfB.get_B_map(s_shape, s_mean, u, alpha, h, min_s_0, r_distance_array, r_array, exon_array, window_size)

		# print("epoch0", min_s_0, flush=True)
		# print("epoch0", B_array_0, flush=True)

		if B_array_0 is None:
			if maximize:
				return float("-inf")
			else:
				return float("inf")

		rs_iters += 1

		thresh_diff = abs(min_s_0 - prev_thresh) / min_s_0

		prev_thresh = min_s_0

	prev_thresh = min_s_1
	thresh_diff = float("infinity")
	rs_iters = 0

	while thresh_diff > max_thresh_diff and rs_iters < max_rs_iters:

		min_s_1 = 3 / (np.mean(B_array_1) * 2 * Ne1 / (1 + F)) / (h*(1 - F) + F)
		B_array_1 = BinfB.get_B_map(s_shape, s_mean, u, alpha, h, min_s_1, r_distance_array, r_array, exon_array, window_size)

		# print("epoch1", min_s_1, flush=True)
		# print("epoch1", B_array_1, flush=True)

		if B_array_1 is None:
			if maximize:
				return float("-inf")
			else:
				return float("inf")

		rs_iters += 1

		thresh_diff = abs(min_s_1 - prev_thresh) / min_s_1

		prev_thresh = min_s_1


	prev_thresh = min_s_2
	thresh_diff = float("infinity")
	rs_iters = 0

	while thresh_diff > max_thresh_diff and rs_iters < max_rs_iters:

		min_s_2 = 3 / (np.mean(B_array_2) * 2 * Ne2 / (1 + F)) / (h*(1 - F) + F)
		B_array_2 = BinfB.get_B_map(s_shape, s_mean, u, alpha, h, min_s_2, r_distance_array, r_array, exon_array, window_size)

		if B_array_2 is None:
			if maximize:
				return float("-inf")
			else:
				return float("inf")

		rs_iters += 1

		thresh_diff = abs(min_s_2 - prev_thresh) / min_s_2

		prev_thresh = min_s_2

	### rescaling done

	try:
		ND_step_1, ND_step_2 = BinfB.get_ND_two_step(s_shape, s_mean * (h*(1 - F) + F), min_s_0)
	except:
		if maximize:
			return float("-inf")
		else:
			return float("inf")

	SFS_lookup_BGS = {}

	for idx, window_SFS in enumerate(observed_SFS_windows):

		if np.sum(window_SFS) == 0:
			pass

		else:

			if precision == 0:

				SFS_key = tuple([round(B_array_0[idx], 2), round(B_array_1[idx], 2), round(B_array_2[idx], 2)])

				if SFS_key not in SFS_lookup_BGS.keys():
					try:
						if alpha == None:
							SFS_lookup_BGS[SFS_key] = BinfS.get_4N_SFS_ND(ND_step_1, ND_step_2, step0, step1, step1 * 1.1, 
								Ne0, Ne1, Ne2, Ne2, round(B_array_0[idx], 2), round(B_array_1[idx], 2), 
								round(B_array_2[idx], 2), round(B_array_2[idx], 2), mu, genomes)
						else:
							SFS_lookup_BGS[SFS_key] = BinfS.get_2N_SFS_selfing_ND(ND_step_1, ND_step_2, step, Ne0, Ne1, 
								round(B_array_0[idx], 2), round(B_array_1[idx], 2), alpha, mu, genomes)

						if (SFS_lookup_BGS[SFS_key] <= 0).sum() > 0 or (SFS_lookup_BGS[SFS_key] >= 1).sum() > 0 or np.isnan(np.sum(SFS_lookup_BGS[SFS_key])):
							if maximize:
								return float("-inf")
							else:
								return float("inf")
						expected_SFS = SFS_lookup_BGS[SFS_key]

					except:
						if maximize:
							return float("-inf")
						else:
							return float("inf")

				else:
					expected_SFS = SFS_lookup_BGS[SFS_key]


			elif precision == 1:

				SFS_key = tuple([round(B_array_0[idx], 3), round(B_array_1[idx], 3), round(B_array_2[idx], 3)])

				if SFS_key not in SFS_lookup_BGS.keys():
					try:
						if alpha == None:
							SFS_lookup_BGS[SFS_key] = BinfS.get_4N_SFS_ND(ND_step_1, ND_step_2, step0, step1, step1 * 1.1, 
								Ne0, Ne1, Ne2, Ne2, round(B_array_0[idx], 3), round(B_array_1[idx], 3), 
								round(B_array_2[idx], 3), round(B_array_2[idx], 3), mu, genomes)
						else:
							SFS_lookup_BGS[SFS_key] = BinfS.get_2N_SFS_selfing_ND(ND_step_1, ND_step_2, step, Ne0, Ne1, 
								round(B_array_0[idx], 3), round(B_array_1[idx], 3), alpha, mu, genomes)

						if (SFS_lookup_BGS[SFS_key] <= 0).sum() > 0 or (SFS_lookup_BGS[SFS_key] >= 1).sum() > 0 or np.isnan(np.sum(SFS_lookup_BGS[SFS_key])):
							if maximize:
								return float("-inf")
							else:
								return float("inf")
						expected_SFS = SFS_lookup_BGS[SFS_key]
					except:
						if maximize:
							return float("-inf")
						else:
							return float("inf")

				else:
					expected_SFS = SFS_lookup_BGS[SFS_key]


			else:
				try:
					if alpha == None:
						expected_SFS = BinfS.get_4N_SFS_ND(ND_step_1, ND_step_2, step0, step1, step1 * 1.1, 
								Ne0, Ne1, Ne2, Ne2, B_array_0[idx], B_array_1[idx], 
								B_array_2[idx], B_array_2[idx], mu, genomes)
					else:
						expected_SFS = BinfS.get_2N_SFS_selfing_ND(ND_step_1, ND_step_2, step, Ne0, Ne1, 
							B_array_0[idx], B_array_1[idx], alpha, mu, genomes)

					if (expected_SFS <= 0).sum() > 0 or (expected_SFS >= 1).sum() > 0 or np.isnan(np.sum(expected_SFS)):
						if maximize:
							return float("-inf")
						else:
							return float("inf")
				except:
					if maximize:
						return float("-inf")
					else:
						return float("inf")

			if eps:
				expected_SFS = polarisation_error(expected_SFS, eps, 1)

			window_SFS_copy = np.copy(window_SFS)

			if mask_hf:
				expected_SFS[0] += expected_SFS[-1]
				expected_SFS[0] += expected_SFS[-2]
				expected_SFS = expected_SFS[0:genomes-2]

				window_SFS_copy[0] += window_SFS_copy[-1]
				window_SFS_copy[0] += window_SFS_copy[-2]
				window_SFS_copy = window_SFS_copy[0:genomes-2]


			log_expected_SFS = np.log(expected_SFS) # this will create a warning if there is a zero in there
			temp_likelihoods = np.multiply(log_expected_SFS, window_SFS_copy) # this will lead to -inf*0 -> nan, or -inf*[>0] -> -inf, nan is fine!
			loglikelihood += np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)) # change nan to zero

			if loglikelihood >= 0:
				print(expected_SFS)
				print(log_expected_SFS)
				print(temp_likelihoods)
				print(np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)))
				sys.exit()

	print(params, loglikelihood, flush=True)
	if maximize:
		return loglikelihood
	else:
		return -1 * loglikelihood

def do_calc(sfs_chunk):
	results_chunk = []
	for sfs_c in sfs_chunk:
		sfs_dict = {}
		ND_step_1, ND_step_2, step0, step1, step2, Ne0, Ne1, Ne2, Ne3, B0, B1, B2, B2, mu, genomes, idx = sfs_c
		try:
			sfs_dict["sfs"] = BinfS.get_4N_SFS_ND(ND_step_1, ND_step_2, step0, step1, step2, Ne0, Ne1, Ne2, Ne3, B0, B1, B2, B2, mu, genomes)
		except:
			sfs_dict["sfs"] = None
		if (sfs_dict["sfs"] <= 0).sum() > 0 or (sfs_dict["sfs"] >= 1).sum() > 0 or np.isnan(np.sum(sfs_dict["sfs"])):
			sfs_dict["sfs"] = None
		sfs_dict["idx"] = idx
		results_chunk.append(sfs_dict)
	return results_chunk

def rescale_min_s(arg_list):

	max_thresh_diff, max_rs_iters, s_shape, s_mean, u, alpha, h, min_s, r_distance_array, r_array, exon_array, F, alpha, Ne = arg_list

	B_array = BinfB.get_B_map(s_shape, s_mean, u, alpha, h, min_s, r_distance_array, r_array, exon_array, window_size)

	if B_array is None:
		#print("B-map failed #0")
		return None, None

	prev_thresh = min_s
	thresh_diff = float("infinity")
	rs_iters = 0

	while thresh_diff > max_thresh_diff and rs_iters < max_rs_iters:

		min_s = 3 / (np.mean(B_array) * 2 * Ne / (1 + F)) / (h*(1 - F) + F)
		B_array = BinfB.get_B_map(s_shape, s_mean, u, alpha, h, min_s, r_distance_array, r_array, exon_array, window_size)

		#print("min_s:", min_s, flush=True)

		if B_array is None:
			#print("B-map failed #1")
			return None, None

		rs_iters += 1

		thresh_diff = abs(min_s - prev_thresh) / min_s

		prev_thresh = min_s

	if thresh_diff > max_thresh_diff and max_rs_iters > 1:
		return [None, None]

	return [B_array, min_s]

def do_calc_selfing(sfs_chunk):
	results_chunk = []
	for sfs_c in sfs_chunk:
		sfs_dict = {}
		ND_step_1, ND_step_2, step, N0, N1, B0, B1, alpha, mu, genomes, idx = sfs_c
		#print(ND_step_1, ND_step_2, step, N0, N1, B0, B1, alpha, mu, genomes, idx, flush=True)
		try:
			sfs_dict["sfs"] = BinfS.get_2N_SFS_selfing_ND(ND_step_1, ND_step_2, step, N0, N1, B0, B1, alpha, mu, genomes)
		except:
			#print("sfs calculation failed", flush=True)
			sfs_dict["sfs"] = None
		if (sfs_dict["sfs"] <= 0).sum() > 0 or (sfs_dict["sfs"] >= 1).sum() > 0 or np.isnan(np.sum(sfs_dict["sfs"])):
			#print("sfs is malformed", flush=True)
			#print(sfs_dict["sfs"], flush=True)
			sfs_dict["sfs"] = None
		sfs_dict["idx"] = idx
		results_chunk.append(sfs_dict)
	return results_chunk

def optimisation_3N_classicBGS_mp(params, grad, mu, observed_SFS_windows, verbose, 
	r_distance_array, exon_array, r_array, mode, ploidy, genomes, observed_Ne, maximize, precision, alpha, 
	mask_hf, eps, processes, time_test):

	if alpha:
		sys.exit("not implemented")

	if len(params) == 8:
		Ne0 = params[0]
		Ne1 = params[1]
		Ne2 = params[2]
		step0 = params[3]
		step1 = params[4]
		u = params[5]
		s_mean = params[6]
		s_shape = params[7]
		h = 0.5

	loglikelihood = 0

	if alpha:
		F = alpha / (2 - alpha)
	else:
		F = 0

	if step0 > step1:
		if maximize:
			return float("-inf")
		else:
			return float("inf")

	overall_Nmax = overall_Ne_recursion(0, 0, 0, [Ne0, step0, Ne1, step1, Ne2, float("inf")], 0)

	approx_reduction = observed_Ne / overall_Nmax

	min_s_0 = 3 / (approx_reduction * 2 * Ne0) / (h*(1 - F) + F)
	min_s_1 = 3 / (approx_reduction * 2 * Ne1) / (h*(1 - F) + F)
	min_s_2 = 3 / (approx_reduction * 2 * Ne2) / (h*(1 - F) + F)

	if precision == 0:
		max_rs_iters = 1
		max_thresh_diff = 0.1
	elif precision == 1:
		max_rs_iters = 5
		max_thresh_diff = 0.01
	else:
		max_rs_iters = 10
		max_thresh_diff = 0.001

	arg_list_list = []

	arg_list_list.append([max_thresh_diff, max_rs_iters, s_shape, s_mean, u, alpha, h, min_s_0, r_distance_array, r_array, exon_array, F, 0, Ne0])
	arg_list_list.append([max_thresh_diff, max_rs_iters, s_shape, s_mean, u, alpha, h, min_s_1, r_distance_array, r_array, exon_array, F, 0, Ne1])
	arg_list_list.append([max_thresh_diff, max_rs_iters, s_shape, s_mean, u, alpha, h, min_s_2, r_distance_array, r_array, exon_array, F, 0, Ne2])

	results = []
	with poolcontext(processes=min(processes, 3)) as pool:
		if time_test:
			t0 = time.time()
		for result in pool.imap_unordered(rescale_min_s, arg_list_list, 1):
			results += result
			if time_test:
				t1 = time.time()
				print(t1 - t0)


	B_array_0, min_s_0, B_array_1, min_s_1, B_array_2, min_s_2 = results

	if B_array_0 is None or B_array_1 is None or B_array_2 is None:
		if maximize:
			return float("-inf")
		else:
			return float("inf")

	### rescaling done

	try:
		ND_step_1, ND_step_2 = BinfB.get_ND_two_step(s_shape, s_mean * (h*(1 - F) + F), min_s_0)
	except:
		if maximize:
			return float("-inf")
		else:
			return float("inf")

	sfs_params = []

	for idx, window_SFS in enumerate(observed_SFS_windows):

		if np.sum(window_SFS) == 0:
			pass

		else:
			sfs_params.append([ND_step_1, ND_step_2, step0, step1, step1 * 1.1, Ne0, Ne1, Ne2, Ne2, B_array_0[idx], B_array_1[idx], B_array_2[idx], B_array_2[idx], mu, genomes, idx])

	results = []

	sfs_params_chunked = [sfs_params_chunk for sfs_params_chunk in chunks(sfs_params, math.ceil(len(sfs_params) / (processes*4)))]
	with poolcontext(processes=processes) as pool:
		if time_test:
			t0 = time.time()
		for result in pool.imap_unordered(do_calc, sfs_params_chunked, 1):
			results += result
			if time_test:
				t1 = time.time()
				print(t1 - t0)

	for result in results:
		if result["sfs"] is None:
			if maximize:
				return float("-inf")
			else:
				return float("inf")
		else:
			if eps:
				expected_SFS = polarisation_error(result["sfs"], eps, 1)

			window_SFS_copy = np.copy(observed_SFS_windows[result["idx"]])

			if mask_hf:
				expected_SFS[0] += expected_SFS[-1]
				expected_SFS[0] += expected_SFS[-2]
				expected_SFS = expected_SFS[0:genomes-2]

				window_SFS_copy[0] += window_SFS_copy[-1]
				window_SFS_copy[0] += window_SFS_copy[-2]
				window_SFS_copy = window_SFS_copy[0:genomes-2]

			log_expected_SFS = np.log(expected_SFS) # this will create a warning if there is a zero in there
			temp_likelihoods = np.multiply(log_expected_SFS, window_SFS_copy) # this will lead to -inf*0 -> nan, or -inf*[>0] -> -inf, nan is fine!
			loglikelihood += np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)) # change nan to zero

			if loglikelihood >= 0:
				print(expected_SFS)
				print(log_expected_SFS)
				print(temp_likelihoods)
				print(np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)))
				sys.exit()

	print(params, loglikelihood, flush=True)
	if maximize:
		return loglikelihood
	else:
		return -1 * loglikelihood


def optimisation_1N_classicBGS_selfing(params, grad, mu, observed_SFS_windows, verbose, 
	r_distance_array, exon_array, r_array, mode, ploidy, genomes, observed_Ne, maximize, precision, alpha, 
	mask_hf, eps, window_size):

	if alpha is None:
		sys.exit("not implemented")

	if len(params) == 4:
		Ne0 = params[0]
		u = params[1]
		s_mean = params[2]
		s_shape = params[3]
		h = 0.5

	elif len(params) == 5:
		Ne0 = params[0]
		u = params[1]
		s_mean = params[2]
		s_shape = params[3]
		alpha = params[4]
		h = 0.5

	#print(params, flush=True)

	loglikelihood = 0

	if alpha:
		F = alpha / (2 - alpha)
	else:
		F = 0

	diploids = genomes // 2
	self_2Ne = (2 * observed_Ne) * ((1 - 2*diploids) / (1 + diploids*(-2 + alpha)))

	min_s_0 = 3 / self_2Ne / (h*(1 - F) + F)

	B_array_0 = BinfB.get_B_map(s_shape, s_mean, u, alpha, h, min_s_0, r_distance_array, r_array, exon_array, window_size)

	if B_array_0 is None:
		print(params, "B-map calculation failed", flush=True)
		if maximize:
			return float("-inf")
		else:
			return float("inf")

	prev_thresh = min_s_0
	thresh_diff = float("infinity")
	rs_iters = 0

	if mode == "SFS":

		if np.mean(B_array_0) < 0.001:
			print(params, "B-map mean < 0.001", flush=True)
			if maximize:
				return float("-inf")
			else:
				return float("inf")

	try:
		ND_step_1, ND_step_2 = BinfB.get_ND_two_step(s_shape, s_mean * (h*(1 - F) + F), min_s_0)
	except:
		if maximize:
			return float("-inf")
		else:
			return float("inf")

	if mode == "SFS":

		sfs_dict = {}

		for idx, window_SFS in enumerate(observed_SFS_windows):

			if np.sum(window_SFS) == 0:
				pass

			else:
				if precision == 0:
					sfs_key = tuple([round(B_array_0[idx], 2)])
				elif precision == 1:
					sfs_key = tuple([round(B_array_0[idx], 3)])
				if (precision == 0 or precision == 1) and sfs_key in sfs_dict.keys():
					expected_SFS = sfs_dict[sfs_key]
				else:
					try:
						if precision == 0:
							sfs_dict[sfs_key] = BinfS.get_2N_SFS_selfing_ND(ND_step_1, ND_step_2, Ne0*1.1, Ne0, Ne0, round(B_array_0[idx], 2), round(B_array_0[idx], 2), alpha, mu, genomes)
							expected_SFS = sfs_dict[sfs_key]
						elif precision == 1:
							sfs_dict[sfs_key] = BinfS.get_2N_SFS_selfing_ND(ND_step_1, ND_step_2, Ne0*1.1, Ne0, Ne0, round(B_array_0[idx], 3), round(B_array_0[idx], 3), alpha, mu, genomes)
							expected_SFS = sfs_dict[sfs_key]
						else:
							expected_SFS = BinfS.get_2N_SFS_selfing_ND(ND_step_1, ND_step_2, Ne0*1.1, Ne0, Ne0, B_array_0[idx], B_array_0[idx], alpha, mu, genomes)
					except:
						if maximize:
							return float("-inf")
						else:
							return float("inf")
					if (expected_SFS <= 0).sum() > 0 or (expected_SFS >= 1).sum() > 0 or np.isnan(np.sum(expected_SFS)):
						if maximize:
							return float("-inf")
						else:
							return float("inf")
				
				if eps:
					expected_SFS = polarisation_error(expected_SFS, eps, 1)

				window_SFS_copy = np.copy(window_SFS)

				if mask_hf:
					expected_SFS[0] += expected_SFS[-1]
					expected_SFS[0] += expected_SFS[-2]
					expected_SFS = expected_SFS[0:genomes-2]

					window_SFS_copy[0] += window_SFS_copy[-1]
					window_SFS_copy[0] += window_SFS_copy[-2]
					window_SFS_copy = window_SFS_copy[0:genomes-2]

				log_expected_SFS = np.log(expected_SFS) # this will create a warning if there is a zero in there
				temp_likelihoods = np.multiply(log_expected_SFS, window_SFS_copy) # this will lead to -inf*0 -> nan, or -inf*[>0] -> -inf, nan is fine!
				loglikelihood += np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)) # change nan to zero

				if loglikelihood >= 0:
					print(expected_SFS)
					print(log_expected_SFS)
					print(temp_likelihoods)
					print(np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)))
					sys.exit()

	elif mode == "pi":

		E_pi = np.array([overall_Ne_recursion(0, 0, 0, [Ne0, ND_step_1, (Ne0 + Ne0*B) / 2, ND_step_2, Ne0*B, float("inf")], 0) * 4 * mu / ((1 - 2*diploids) / (1 + diploids*(-2 + alpha))) for B in B_array_0])
		P_mismatch = np.array([np.array([e_pi / (1 + e_pi), 1 - (e_pi / (1 + e_pi))]) for e_pi in E_pi])

		for idx, window_SFS in enumerate(observed_SFS_windows):

			if np.sum(window_SFS) == 0:
				pass

			else:

				# probably better to do this once, outside of optimisation
				focal_window_mm = SFS_to_pi(window_SFS, genomes)

				log_expected_mm = np.log(P_mismatch[idx]) # this will create a warning if there is a zero in there
				temp_likelihoods = np.multiply(log_expected_mm, focal_window_mm) # this will lead to -inf*0 -> nan, or -inf*[>0] -> -inf, nan is fine!
				loglikelihood += np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)) # change nan to zero

	print(params, loglikelihood, flush=True)
	if maximize:
		return loglikelihood
	else:
		return -1 * loglikelihood


def optimisation_1N_classicBGS_selfing_mp(params, grad, mu, observed_SFS_windows, verbose, 
	r_distance_array, exon_array, r_array, mode, ploidy, genomes, observed_Ne, maximize, precision, alpha, 
	mask_hf, eps, processes, window_size):

	if alpha is None:
		sys.exit("not implemented")

	if len(params) == 4:
		Ne0 = params[0]
		u = params[1]
		s_mean = params[2]
		s_shape = params[3]
		h = 0.5

	elif len(params) == 5:
		Ne0 = params[0]
		u = params[1]
		s_mean = params[2]
		s_shape = params[3]
		alpha = params[4]
		h = 0.5

	#print(params, flush=True)

	loglikelihood = 0

	if alpha:
		F = alpha / (2 - alpha)
	else:
		F = 0

	diploids = genomes // 2
	self_2Ne = (2 * observed_Ne) * ((1 - 2*diploids) / (1 + diploids*(-2 + alpha)))

	min_s_0 = 3 / self_2Ne / (h*(1 - F) + F)

	B_array_0 = BinfB.get_B_map(s_shape, s_mean, u, alpha, h, min_s_0, r_distance_array, r_array, exon_array, window_size)

	if B_array_0 is None:
		#print("B-map failed", flush=True)
		if maximize:
			return float("-inf")
		else:
			return float("inf")

	# dadi is super slow when there are recent dramatic changes in Ne
	# this line limits this
	if np.mean(B_array_0) < 0.001:
		if maximize:
			return float("-inf")
		else:
			return float("inf")

	try:
		ND_step_1, ND_step_2 = BinfB.get_ND_two_step(s_shape, s_mean * (h*(1 - F) + F), min_s_0)
	except:
		if maximize:
			return float("-inf")
		else:
			return float("inf")

	sfs_params = []

	for idx, window_SFS in enumerate(observed_SFS_windows):

		if np.sum(window_SFS) == 0:
			pass

		else:
			sfs_params.append([ND_step_1, ND_step_2, ND_step_2*2, Ne0, Ne0, B_array_0[idx], B_array_0[idx], alpha, mu, genomes, idx])

	results = []

	sfs_params_chunked = [sfs_params_chunk for sfs_params_chunk in chunks(sfs_params, math.ceil(len(sfs_params) / (processes*4)))]
	with poolcontext(processes=processes) as pool:
		#t0 = time.time()
		for result in pool.imap_unordered(do_calc_selfing, sfs_params_chunked, 1):
			results += result
			#t1 = time.time()
			#print(t1 - t0, flush=True)

	for result in results:
		if result["sfs"] is None:
			if maximize:
				return float("-inf")
			else:
				return float("inf")
		else:
			if eps:
				expected_SFS = polarisation_error(result["sfs"], eps, 1)

			window_SFS_copy = np.copy(observed_SFS_windows[result["idx"]])

			if mask_hf:
				expected_SFS[0] += expected_SFS[-1]
				expected_SFS[0] += expected_SFS[-2]
				expected_SFS = expected_SFS[0:genomes-2]

				window_SFS_copy[0] += window_SFS_copy[-1]
				window_SFS_copy[0] += window_SFS_copy[-2]
				window_SFS_copy = window_SFS_copy[0:genomes-2]

			log_expected_SFS = np.log(expected_SFS) # this will create a warning if there is a zero in there
			temp_likelihoods = np.multiply(log_expected_SFS, window_SFS_copy) # this will lead to -inf*0 -> nan, or -inf*[>0] -> -inf, nan is fine!
			loglikelihood += np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)) # change nan to zero

			if loglikelihood >= 0:
				print(expected_SFS)
				print(log_expected_SFS)
				print(temp_likelihoods)
				print(np.sum(np.where(np.isnan(temp_likelihoods), 0, temp_likelihoods)))
				sys.exit()

	print(params, loglikelihood, flush=True)
	if maximize:
		return loglikelihood
	else:
		return -1 * loglikelihood


def overall_Ne_recursion(overall_Ne, coal_already, idx, epochs, time):
	Ne = epochs[idx]
	T = epochs[idx+1]
	coal_prob = (1 - coal_already) * (1 - np.exp(-(T-time)/(2*Ne)))
	coal_already += coal_prob
	overall_Ne += Ne * coal_prob
	time += (T - time)
	if idx != len(epochs) - 2:
		idx += 2
		return overall_Ne_recursion(overall_Ne, coal_already, idx, epochs, time)
	return overall_Ne