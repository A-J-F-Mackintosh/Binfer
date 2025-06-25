#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Usage: Binfer.py -s <STR> -b <STR> -g <STR> -r <STR> -p <INT> -n <INT> -w <INT> -m <FLT> [-h -f -u -a -y <FLT> -q <INT>]

  [Options]
    -s, --sfs <STR>                            SFS file
    -b, --bed <STR>                            Bed file of sites under purifying selection
    -g, --genomefile <STR>                     Genomefile of sequence lengths
    -r, --rmaps <STR>                          Recombination maps
    -p, --ploidy <INT>                         Ploidy, 1 for haploid and 2 for diploid
    -n, --individuals <INT>                    Number of individuals in sample
    -w, --windowsize <INT>                     Bases in each window
    -m, --mu <FLT>                             De novo mutation rate per-site per-generation
    -f, --selfing                              Fit models with partial selfing
    -u, --fixu                                 Fix the deleterious mutation rate to the de novo rate
    -a, --mask                                 Mask last two SFS entries which are prone to reference bias
    -y, --eps <FLT>                            Use a specific polarisation error rate
    -q, --processes <INT>                      Number of CPUs to use (default is 1)
    -h, --help                                 Show this message

"""


from docopt import docopt
import sys
import os
import time
import random
import nlopt
from functools import partial
import math
import numba as nb
import numpy as np
import moments
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum
import scipy.integrate as integrate
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("default", category=DeprecationWarning)
np.set_printoptions(linewidth=np.inf)

import contextlib
import multiprocessing

############################################### r-map functions ###############################################

@nb.njit
def find_window(start, idx, window_size):
	window_start = idx * window_size
	window_end = window_start + window_size
	if window_start <= start < window_end:
		return idx
	else:
		return find_window(start, idx+1, window_size)


def pmap_to_rmap(p_pos, r_map, ignore_breaks):
	new_pos = []
	for pos in p_pos:
		total_rec = 0
		for rec in r_map:
			start, end, M = rec
			if ignore_breaks and M == 0.5 and end - start == 1:
				pass
			elif pos >= end:
				if M < 0.5:
					total_rec += M
				else:
					total_rec += 100 # this ensures that the mapping function will return 0.5
			else:
				total_rec += ((pos - start) / (end - start)) * M
				break
		new_pos.append(total_rec) # Morgans

	return new_pos

def r_map_to_chrom_map(r_map, prec):
	prec = float(prec)
	if prec != float("inf"):
		prec = prec * 1e6
	chrom_map = []
	total_r = 0
	bases_so_far = 0
	for entry in r_map:
		if entry[1] - entry[0] > prec:
			sys.exit("map window size is too small")
		if entry[2] != 0.5:
			if bases_so_far + entry[1] - entry[0] > prec:
				r_contrib = entry[2] * ((prec - bases_so_far) / (entry[1] - entry[0]))
				r_contrib_next = entry[2] - r_contrib
				total_r += r_contrib
				if len(chrom_map) > 0:
					chrom_map.append((chrom_map[-1][1], chrom_map[-1][1] + prec, total_r))
				else:
					chrom_map.append((0, prec, total_r))
				bases_so_far = (entry[1] - entry[0]) - (prec - bases_so_far)
				total_r = r_contrib_next
			else:
				bases_so_far += entry[1] - entry[0]
				total_r += entry[2]
		else:
			if len(chrom_map) > 0:
				chrom_map.append((chrom_map[-1][1], entry[1], total_r))
			else:
				chrom_map.append((0, entry[1], total_r))
			chrom_map.append(entry)
			total_r = 0
			bases_so_far = 0
	return chrom_map

########## arrays used for interpolation ###############

r_inter = np.array([5e-5, 1e-5, 
	2.5e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 
	2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 
	2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 
	2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 0.1, 
	0.2, 0.3, 0.4, 0.5])

r_inter_within = np.array([5e-10, 7.5e-10, 1e-9, 2.5e-9, 5e-9, 7.5e-9, 
	1e-8, 2.5e-8, 5e-8, 7.5e-8, 1e-7]) * 10_000



################################################ main ################################################

if __name__ == '__main__':
	__version__ = '0.1'
	args = docopt(__doc__)

	print("[+] Reading args...", flush=True)

	print(args)

	sfs_f = args["--sfs"]
	bed_f = args["--bed"]
	genome_f = args["--genomefile"]
	rmap_f = args["--rmaps"]
	if args["--mask"]:
		mask_hf = True
	else:
		mask_hf = False
	f_eps = args["--eps"]
	if f_eps:
		f_eps = float(f_eps)

	window_size = int(args["--windowsize"])
	ploidy = int(args["--ploidy"])
	ind = int(args["--individuals"])
	mu = float(args["--mu"])

	genomes = ploidy*ind

	SFS_array = np.genfromtxt(sfs_f, delimiter=",")
	if args["--selfing"]:
		Fis = float(np.genfromtxt(sfs_f.replace("sfs", "fis"), delimiter=","))
		alpha = (2*Fis) / (1 + Fis)
	else:
		alpha = None

	if args["--processes"]:
		processes = int(args["--processes"])
	else:
		processes = 1


	import likelihood as BinfL
	import CRS2 as BinfC


	print("[+] Reading genome file...", flush=True)

	target_sequences = []
	windows = 0
	seq_to_pos = {} # gives number of nucleotide needed to add to a bed pos to give its array pos

	with open(genome_f, "r") as file:
		for line in file:
			target_seq, target_length = line.split("\t")[0:2]
			target_sequences.append(target_seq)
			seq_to_pos[target_seq] = windows * window_size
			target_windows = int(math.ceil(float(target_length) / window_size))
			windows += target_windows

	if windows != SFS_array.shape[0]:
		sys.exit("Number of windows implied by genomefile ({}) \
			does not match the SFS file ({})".format(windows, SFS_array.shape[0]))

	print("[=] Read {} sequences and {} windows".format(len(target_sequences), windows), flush=True)


	print("[+] Reading bed file...", flush=True)

	exon_array = np.zeros(windows, dtype=int)

	intervals = 0
	window_idx = 0

	with open(bed_f, "r") as file:
		for line in file:
			intervals += 1
			seq, start, end = line.split("\t")[0:3]

			start = int(start) + seq_to_pos[seq]
			end = int(end) + seq_to_pos[seq]

			window_idx = find_window(start, window_idx, window_size)

			contrib_to_window = min(((window_idx + 1) * window_size), end) - start
			exon_array[window_idx] += contrib_to_window

			if end > ((window_idx + 1) * window_size):
				looking_for_end = True

				while looking_for_end:
					window_idx += 1
					if end <= ((window_idx + 1) * window_size):
						contrib_to_window = end - (window_idx * window_size) 
						exon_array[window_idx] += contrib_to_window
						looking_for_end = False
					else:
						exon_array[window_idx] += window_size

	print("[=] Read {} intervals".format(intervals), flush=True)


	print("[+] Reading in recombination map...", flush=True)

	rmap_files = []

	with open(rmap_f, "r") as file:
		for line in file:
			rmap_files.append(line.split()[0])


	r_map = [] # tuples of coords with Morgan span
	start = float(0)
	so_far = float(0)

	for rmap_file in rmap_files:
		with open(rmap_file, "r") as r_file: # cM / Mb rate in windows
			for line in r_file:
				if line.startswith("phys"):
					pass
				else:
					phys, rec, upper, lower = line.split("\t")
					end = (float(phys) * 1e6) + so_far
					rec = float(rec) * 1e-8
					#r_vals.append(rec)
					r_map.append((start, end, rec * (end - start)))
					start = end
			r_map[-1] = (r_map[-1][0], r_map[-1][1] - 1, r_map[-1][2])
			r_map.append((r_map[-1][1], end, 0.5))
			so_far = end + 1

	chromosome_end = window_size * len(SFS_array)
	r_array = np.zeros(windows, dtype=float)

	# per-base recombination rate in each window
	for w in range(0, windows):
		w_r_dist = pmap_to_rmap([w * window_size, (w + 1) * window_size], r_map, ignore_breaks = True)
		r_array[w] = ((w_r_dist[1] - w_r_dist[0])) / window_size
		#print(r_array[w])

	r_windows_mid = np.array(pmap_to_rmap([(w * window_size) + (window_size / 2.0) for w in range(0, windows)], r_map, ignore_breaks = False))
	r_distance_array = np.array([np.array([(1 - math.exp(-2*abs(window_pos - mid))) / 2 for mid in r_windows_mid]) \
		for window_pos in r_windows_mid])



	if args["--selfing"]:
		print("[=] Selfing rate (alpha) = {}".format(alpha))
	

	print("[+] Optimising 1N pi model...", flush=True)
	opt = nlopt.opt(nlopt.LN_NELDERMEAD, 1)
	opt.set_maxeval(3000)
	opt.set_upper_bounds([5_000_000])
	opt.set_lower_bounds([10])
	nlopt_function = partial(BinfL.optimisation_1N, mu = mu, observed_SFS = BinfL.SFS_to_pi(np.sum(SFS_array, axis=0), genomes), 
		verbose = True, mode = "pi", genomes = genomes)
	opt.set_max_objective(nlopt_function)
	xopt = opt.optimize([10_000])
	print("[=] Ne: {}".format(xopt[0]), flush=True)
	print("[=] Loglikelihood: {}".format(opt.last_optimum_value()), flush=True)
	single_Ne_estimate = xopt[0]


	if args["--selfing"] and ploidy == 2:

		print("[+] Optimising 2N model with selfing...", flush=True)

		if f_eps: # if polarisation error set in args

			opt = nlopt.opt(nlopt.LN_NELDERMEAD, 3)
			opt.set_maxeval(2500)
			# two params, ratio of current to anc, and ratio of step to 2*anc, log10 them for better search
			opt.set_upper_bounds([math.log10(100), math.log10(100), math.log10(0.9999)])
			opt.set_lower_bounds([math.log10(0.01), math.log10(0.01), math.log10(0.0001)])
			nlopt_function = partial(BinfL.optimisation_multi_epoch, mu = mu, observed_SFS = np.sum(SFS_array, axis=0), 
				verbose = True, genomes = genomes, epochs = 2, alpha = alpha, mask_hf = mask_hf, eps = f_eps)
			opt.set_max_objective(nlopt_function)
			xopt = opt.optimize([0, 0, math.log10(0.5)])
			N_ratio, step_ratio, alpha = xopt

			alpha = 10**alpha
			
			opt = nlopt.opt(nlopt.LN_NELDERMEAD, 1)
			opt.set_maxeval(1500)
			opt.set_upper_bounds([math.log10(0.5)])
			opt.set_lower_bounds([math.log10(1e-7)])
			nlopt_function = partial(BinfL.optimisation_theta, observed_SFS = np.sum(SFS_array, axis=0), 
				other_params = [N_ratio, step_ratio, f_eps], genomes = genomes, epochs = 2, alpha = alpha, 
				mask_hf = mask_hf)
			opt.set_max_objective(nlopt_function)
			xopt = opt.optimize([math.log10(0.01)])
			theta_2N = xopt[0]

			Ne1_estimate = (10**theta_2N) / (4 * mu)
			Ne0_estimate = (10**N_ratio) * Ne1_estimate
			step_estimate = (10**step_ratio) * 2 * Ne1_estimate
			print("[=] Ne0, Ne1, T, alpha: {}, {}, {}, {}".format(Ne0_estimate, Ne1_estimate, step_estimate, alpha))
			print("[=] Loglikelihood: {}".format(opt.last_optimum_value()))

			"""
			print("[+] Optimising 3N model with selfing...", flush=True)

			best_lnCL = float("-inf")

			for i in range(0, 8):

				opt = nlopt.opt(nlopt.GN_CRS2_LM, 4)
				opt.set_maxeval(6_000)
				opt.set_population(196)
				# two params, ratio of current to anc, and ratio of step to 2*anc, log10 them for better search
				opt.set_upper_bounds([math.log10(10), math.log10(10), math.log10(10), math.log10(10)])
				opt.set_lower_bounds([math.log10(0.1), math.log10(0.1), math.log10(0.1), math.log10(0.1)])
				nlopt_function = partial(BinfL.optimisation_multi_epoch, mu = mu, observed_SFS = np.sum(SFS_array, axis=0), 
					verbose = True, genomes = genomes, epochs = 3, alpha = alpha, mask_hf = mask_hf, eps = f_eps)
				opt.set_max_objective(nlopt_function)
				xopt = opt.optimize([0, 0, 0, 0])
				N0_ratio, N1_ratio, step0_ratio, step1_ratio = xopt

				opt = nlopt.opt(nlopt.LN_NELDERMEAD, 4)
				opt.set_maxeval(4_000)
				# two params, ratio of current to anc, and ratio of step to 2*anc, log10 them for better search
				opt.set_upper_bounds([math.log10(25), math.log10(25), math.log10(25), math.log10(25)])
				opt.set_lower_bounds([math.log10(0.04), math.log10(0.04), math.log10(0.04), math.log10(0.04)])
				nlopt_function = partial(BinfL.optimisation_multi_epoch, mu = mu, observed_SFS = np.sum(SFS_array, axis=0), 
					verbose = True, genomes = genomes, epochs = 3, alpha = alpha, mask_hf = mask_hf, eps = f_eps)
				opt.set_max_objective(nlopt_function)
				xopt = opt.optimize([N0_ratio, N1_ratio, step0_ratio, step1_ratio])
				N0_ratio, N1_ratio, step0_ratio, step1_ratio = xopt

				print(N0_ratio, N1_ratio, step0_ratio, step1_ratio, opt.last_optimum_value(), flush=True)

				if opt.last_optimum_value() > best_lnCL:
					N0_ratio_est, N1_ratio_est, step0_ratio_est, step1_ratio_est = xopt
					best_lnCL = opt.last_optimum_value()
			
			opt = nlopt.opt(nlopt.LN_NELDERMEAD, 1)
			opt.set_maxeval(1500)
			opt.set_upper_bounds([math.log10(0.5)])
			opt.set_lower_bounds([math.log10(1e-7)])
			nlopt_function = partial(BinfL.optimisation_theta, observed_SFS = np.sum(SFS_array, axis=0), 
				other_params = [N0_ratio_est, N1_ratio_est, step0_ratio_est, step1_ratio_est, f_eps], 
				genomes = genomes, epochs = 3, alpha = alpha, mask_hf = mask_hf)
			opt.set_max_objective(nlopt_function)
			xopt = opt.optimize([math.log10(0.01)])
			theta_2N = xopt[0]

			Ne2_estimate = (10**theta_2N) / (4 * mu)
			Ne0_estimate = (10**N0_ratio_est) * Ne2_estimate
			Ne1_estimate = (10**N1_ratio_est) * Ne2_estimate
			step0_estimate = (10**step0_ratio_est) * 2 * Ne2_estimate
			step1_estimate = (10**step1_ratio_est) * 2 * Ne2_estimate

			print("[=] Ne0, Ne1, Ne2, T0, T1: {}, {}, {}, {}, {}".format(Ne0_estimate, Ne1_estimate, Ne2_estimate, 
				step0_estimate, step1_estimate))
			print("[=] Loglikelihood: {}".format(opt.last_optimum_value()))
			"""

		else: # polarisation parameter is estimated instead

			best_lnCL = float("-inf")

			for i in range(0, 8): # best to perform multiple runs

				opt = nlopt.opt(nlopt.GN_CRS2_LM, 4)
				opt.set_maxeval(2000)
				opt.set_population(100)
				opt.set_upper_bounds([math.log10(25), math.log10(25), math.log10(0.5), math.log10(0.9999)])
				opt.set_lower_bounds([math.log10(0.04), math.log10(0.04), math.log10(0.0001), math.log10(0.0001)])
				nlopt_function = partial(BinfL.optimisation_multi_epoch, mu = mu, observed_SFS = np.sum(SFS_array, axis=0), 
					verbose = True, genomes = genomes, epochs = 2, alpha = alpha, mask_hf = mask_hf, eps = f_eps)
				opt.set_max_objective(nlopt_function)
				xopt = opt.optimize([0, 0, math.log10(0.01), math.log10(0.5)])
				N_ratio, step_ratio, eps, alpha = xopt

				opt = nlopt.opt(nlopt.LN_NELDERMEAD, 4)
				opt.set_maxeval(2000)
				# two params, ratio of current to anc, and ratio of step to 2*anc, log10 them for better search
				opt.set_upper_bounds([math.log10(25), math.log10(25), math.log10(0.5), math.log10(0.9999)])
				opt.set_lower_bounds([math.log10(0.04), math.log10(0.04), math.log10(0.0001), math.log10(0.0001)])
				nlopt_function = partial(BinfL.optimisation_multi_epoch, mu = mu, observed_SFS = np.sum(SFS_array, axis=0), 
					verbose = True, genomes = genomes, epochs = 2, alpha = alpha, mask_hf = mask_hf, eps = f_eps)
				opt.set_max_objective(nlopt_function)
				xopt = opt.optimize([N_ratio, step_ratio, eps, alpha])
				N_ratio, step_ratio, eps, alpha = xopt

				print(N_ratio, step_ratio, eps, alpha, opt.last_optimum_value(), flush=True)

				if opt.last_optimum_value() > best_lnCL:
					N_ratio_est, step_ratio_est, eps_est, alpha_est = xopt
					best_lnCL = opt.last_optimum_value()
			
			opt = nlopt.opt(nlopt.LN_NELDERMEAD, 1)
			opt.set_maxeval(1500)
			opt.set_upper_bounds([math.log10(0.5)])
			opt.set_lower_bounds([math.log10(1e-7)])
			nlopt_function = partial(BinfL.optimisation_theta, observed_SFS = np.sum(SFS_array, axis=0), 
				other_params = [N_ratio_est, step_ratio_est, eps_est, alpha_est], genomes = genomes, epochs = 2, alpha = alpha, 
				mask_hf = mask_hf)
			opt.set_max_objective(nlopt_function)
			xopt = opt.optimize([math.log10(0.01)])
			theta_2N = xopt[0]

			Ne1_estimate = (10**theta_2N) / (4 * mu)
			Ne0_estimate = (10**N_ratio_est) * Ne1_estimate
			step_estimate = (10**step_ratio_est) * 2 * Ne1_estimate
			eps = 10**eps_est
			alpha = 10**alpha_est
			init_eps = eps
			print("[=] Ne0, Ne1, T, eps, alpha: {}, {}, {}, {}, {}".format(Ne0_estimate, Ne1_estimate, step_estimate, eps, alpha))
			print("[=] Loglikelihood: {}".format(opt.last_optimum_value()))

			"""
			print("[+] Optimising 3N model with selfing...", flush=True)

			best_lnCL = float("-inf")

			for i in range(0, 10):

				opt = nlopt.opt(nlopt.GN_CRS2_LM, 5)
				opt.set_maxeval(6_000)
				opt.set_population(196)
				# two params, ratio of current to anc, and ratio of step to 2*anc, log10 them for better search
				opt.set_upper_bounds([math.log10(10), math.log10(10), math.log10(10), math.log10(10), math.log10(0.5)])
				opt.set_lower_bounds([math.log10(0.1), math.log10(0.1), math.log10(0.1), math.log10(0.1), math.log10(0.0001)])
				nlopt_function = partial(BinfL.optimisation_multi_epoch, mu = mu, observed_SFS = np.sum(SFS_array, axis=0), 
					verbose = True, genomes = genomes, epochs = 3, alpha = alpha, mask_hf = mask_hf, eps = f_eps)
				opt.set_max_objective(nlopt_function)
				xopt = opt.optimize([0, 0, 0, 0, math.log10(0.01)])
				N0_ratio, N1_ratio, step0_ratio, step1_ratio, eps = xopt

				opt = nlopt.opt(nlopt.LN_NELDERMEAD, 5)
				opt.set_maxeval(4_000)
				# two params, ratio of current to anc, and ratio of step to 2*anc, log10 them for better search
				opt.set_upper_bounds([math.log10(50), math.log10(50), math.log10(50), math.log10(50), math.log10(0.5)])
				opt.set_lower_bounds([math.log10(0.02), math.log10(0.02), math.log10(0.02), math.log10(0.02), math.log10(0.0001)])
				nlopt_function = partial(BinfL.optimisation_multi_epoch, mu = mu, observed_SFS = np.sum(SFS_array, axis=0), 
					verbose = True, genomes = genomes, epochs = 3, alpha = alpha, mask_hf = mask_hf, eps = f_eps)
				opt.set_max_objective(nlopt_function)
				xopt = opt.optimize([N0_ratio, N1_ratio, step0_ratio, step1_ratio, eps])
				N0_ratio, N1_ratio, step0_ratio, step1_ratio, eps = xopt

				print(N0_ratio, N1_ratio, step0_ratio, step1_ratio, eps, opt.last_optimum_value(), flush=True)

				if opt.last_optimum_value() > best_lnCL:
					N0_ratio_est, N1_ratio_est, step0_ratio_est, step1_ratio_est, eps_est = xopt
					best_lnCL = opt.last_optimum_value()
			
			opt = nlopt.opt(nlopt.LN_NELDERMEAD, 1)
			opt.set_maxeval(1500)
			opt.set_upper_bounds([math.log10(0.5)])
			opt.set_lower_bounds([math.log10(1e-7)])
			nlopt_function = partial(BinfL.optimisation_theta, observed_SFS = np.sum(SFS_array, axis=0), 
				other_params = [N0_ratio_est, N1_ratio_est, step0_ratio_est, step1_ratio_est, eps_est], 
				genomes = genomes, epochs = 3, alpha = alpha, mask_hf = mask_hf)
			opt.set_max_objective(nlopt_function)
			xopt = opt.optimize([math.log10(0.01)])
			theta_2N = xopt[0]

			Ne2_estimate = (10**theta_2N) / (4 * mu)
			Ne0_estimate = (10**N0_ratio_est) * Ne2_estimate
			Ne1_estimate = (10**N1_ratio_est) * Ne2_estimate
			step0_estimate = (10**step0_ratio_est) * 2 * Ne2_estimate
			step1_estimate = (10**step1_ratio_est) * 2 * Ne2_estimate
			eps = 10**eps_est

			print("[=] Ne0, Ne1, Ne2, T0, T1, eps: {}, {}, {}, {}, {}, P{".format(Ne0_estimate, Ne1_estimate, Ne2_estimate, 
				step0_estimate, step1_estimate, eps))
			print("[=] Loglikelihood: {}".format(opt.last_optimum_value()))
			"""

	else: # if random mating


		# here we always estimate eps directly
		print("[+] Optimising 2N model...", flush=True)

		opt = nlopt.opt(nlopt.LN_NELDERMEAD, 3)
		opt.set_maxeval(2500)
		# two params, ratio of current to anc, and ratio of step to 2*anc, log10 them for better search
		opt.set_upper_bounds([math.log10(25), math.log10(25), math.log10(0.5)])
		opt.set_lower_bounds([math.log10(0.04), math.log10(0.04), math.log10(0.0001)])
		nlopt_function = partial(BinfL.optimisation_multi_epoch, mu = mu, observed_SFS = np.sum(SFS_array, axis=0), 
			verbose = True, genomes = genomes, epochs = 2, alpha = None, mask_hf = mask_hf, eps = f_eps)
		opt.set_max_objective(nlopt_function)
		xopt = opt.optimize([0, 0, math.log10(0.01)])
		N_ratio, step_ratio, eps = xopt
		
		opt = nlopt.opt(nlopt.LN_NELDERMEAD, 1)
		opt.set_maxeval(1500)
		opt.set_upper_bounds([math.log10(0.5)])
		opt.set_lower_bounds([math.log10(1e-7)])
		nlopt_function = partial(BinfL.optimisation_theta, observed_SFS = np.sum(SFS_array, axis=0), 
			other_params = [N_ratio, step_ratio, eps], genomes = genomes, epochs = 2, alpha = None, 
			mask_hf = mask_hf)
		opt.set_max_objective(nlopt_function)
		xopt = opt.optimize([math.log10(0.01)])
		theta_2N = xopt[0]

		Ne1_estimate = (10**theta_2N) / (4 * mu)
		Ne0_estimate = (10**N_ratio) * Ne1_estimate
		step_estimate = (10**step_ratio) * 2 * Ne1_estimate
		eps = 10**eps
		init_eps = eps
		print("[=] Ne0, Ne1, T, eps: {}, {}, {}, {}".format(Ne0_estimate, Ne1_estimate, step_estimate, eps))
		print("[=] Loglikelihood: {}".format(opt.last_optimum_value()))


		print("[+] Optimising 3N model...", flush=True)

		best_lnCL = float("-inf")

		for i in range(0, 5):

			opt = nlopt.opt(nlopt.GN_CRS2_LM, 5)
			opt.set_maxeval(5_000)
			opt.set_population(196)
			# two params, ratio of current to anc, and ratio of step to 2*anc, log10 them for better search
			opt.set_upper_bounds([math.log10(10), math.log10(10), math.log10(10), math.log10(10), math.log10(init_eps) + math.log10(5)])
			opt.set_lower_bounds([math.log10(0.1), math.log10(0.1), math.log10(0.1), math.log10(0.1), math.log10(init_eps) - math.log10(5)])
			nlopt_function = partial(BinfL.optimisation_multi_epoch, mu = mu, observed_SFS = np.sum(SFS_array, axis=0), 
				verbose = True, genomes = genomes, epochs = 3, alpha = None, mask_hf = mask_hf, eps = f_eps)
			opt.set_max_objective(nlopt_function)
			xopt = opt.optimize([0, 0, 0, 0, math.log10(init_eps)])
			N0_ratio, N1_ratio, step0_ratio, step1_ratio, eps = xopt

			opt = nlopt.opt(nlopt.LN_NELDERMEAD, 5)
			opt.set_maxeval(3_000)
			# two params, ratio of current to anc, and ratio of step to 2*anc, log10 them for better search
			opt.set_upper_bounds([math.log10(25), math.log10(25), math.log10(25), math.log10(25), math.log10(10**eps*10)])
			opt.set_lower_bounds([math.log10(0.04), math.log10(0.04), math.log10(0.04), math.log10(0.04), math.log10(10**eps/10)])
			nlopt_function = partial(BinfL.optimisation_multi_epoch, mu = mu, observed_SFS = np.sum(SFS_array, axis=0), 
				verbose = True, genomes = genomes, epochs = 3, alpha = None, mask_hf = mask_hf, eps = f_eps)
			opt.set_max_objective(nlopt_function)
			xopt = opt.optimize([N0_ratio, N1_ratio, step0_ratio, step1_ratio, eps])
			N0_ratio, N1_ratio, step0_ratio, step1_ratio, eps = xopt

			#print(N0_ratio, N1_ratio, step0_ratio, step1_ratio, eps, opt.last_optimum_value(), flush=True)

			if opt.last_optimum_value() > best_lnCL:
				N0_ratio_est, N1_ratio_est, step0_ratio_est, step1_ratio_est, eps_est = xopt
				best_lnCL = opt.last_optimum_value()
		
		opt = nlopt.opt(nlopt.LN_NELDERMEAD, 1)
		opt.set_maxeval(1500)
		opt.set_upper_bounds([math.log10(0.5)])
		opt.set_lower_bounds([math.log10(1e-7)])
		nlopt_function = partial(BinfL.optimisation_theta, observed_SFS = np.sum(SFS_array, axis=0), 
			other_params = [N0_ratio_est, N1_ratio_est, step0_ratio_est, step1_ratio_est, eps_est], 
			genomes = genomes, epochs = 3, alpha = None, mask_hf = mask_hf)
		opt.set_max_objective(nlopt_function)
		xopt = opt.optimize([math.log10(0.01)])
		theta_2N = xopt[0]

		Ne2_estimate = (10**theta_2N) / (4 * mu)
		Ne0_estimate = (10**N0_ratio_est) * Ne2_estimate
		Ne1_estimate = (10**N1_ratio_est) * Ne2_estimate
		step0_estimate = (10**step0_ratio_est) * 2 * Ne2_estimate
		step1_estimate = (10**step1_ratio_est) * 2 * Ne2_estimate
		eps = 10**eps_est

		print("[=] Ne0, Ne1, Ne2, T0, T1, eps: {}, {}, {}, {}, {}, {}".format(Ne0_estimate, Ne1_estimate, Ne2_estimate, 
			step0_estimate, step1_estimate, eps))
		print("[=] Loglikelihood: {}".format(opt.last_optimum_value()))
	

	print("Optimising 1N-classic-BGS pi model...", flush=True)

	min_mean_s = 1 / single_Ne_estimate
	upper_boundaries = [single_Ne_estimate * 20, 2e-8, 0.1, 10]
	if args["--selfing"] and alpha < 0.5:
		upper_boundaries[1] = 4e-8
		upper_boundaries[2] = 0.25
	elif args["--selfing"] and alpha >= 0.5:
		upper_boundaries[1] = 6e-8
		upper_boundaries[2] = 0.25
	lower_boundaries = [single_Ne_estimate * 0.99, 0, min_mean_s, 0.1]

	nlopt_function = partial(BinfL.optimisation_1N_classicBGS, grad = None, mu = mu, observed_SFS_windows = SFS_array, 
		verbose = True, r_distance_array = r_distance_array, exon_array = exon_array, r_array = r_array, 
		mode = "pi", ploidy = ploidy, genomes = genomes, observed_Ne = single_Ne_estimate, 
		round_B = False, alpha = None, window_size = window_size)

	if args["--fixu"]: # this option fixes the deleterious mutation rate and gives a big speed up

		distribution_init = np.array(["e", "e", "e"])
		mean_init = np.array([single_Ne_estimate * 2, 0.01, 2])
		sd_init = np.array([1, 1, 1]) # 1 is just a placeholder for exp and uniform distributions

		lower_boundaries = lower_boundaries[:1] + lower_boundaries[2:]
		upper_boundaries = upper_boundaries[:1] + upper_boundaries[2:]

		result = BinfC.CRS2_LM(nlopt_function, lower_boundaries, upper_boundaries, 
			distribution_init, mean_init, sd_init, 100, 50, 500, 1e-7, processes)

		loglikelihood, Ne_estimate_BGS, s_mean_estimate, s_shape_estimate = result

		opt = nlopt.opt(nlopt.LN_NELDERMEAD, 3)
		opt.set_maxeval(300)
		opt.set_upper_bounds(list(np.array(upper_boundaries)*1.00001))
		opt.set_lower_bounds(lower_boundaries)
		opt.set_max_objective(nlopt_function)
		xopt = opt.optimize([Ne_estimate_BGS, s_mean_estimate, s_shape_estimate])
		Ne_estimate_BGS, s_mean_estimate, s_shape_estimate = xopt
		loglikelihood = opt.last_optimum_value()

		print("Ne, s_mean, s_shape: {}, {}, {}".format(Ne_estimate_BGS, s_mean_estimate, s_shape_estimate))
		print("Loglikelihood: {}".format(loglikelihood, flush=True))

	else:

		distribution_init = np.array(["e", "u", "e", "e"])
		mean_init = np.array([single_Ne_estimate * 2, 1e-8, 0.01, 2])
		sd_init = np.array([1, 1, 1, 1]) # 1 is just a placeholder for exp and uniform distributions

		result = BinfC.CRS2_LM(nlopt_function, lower_boundaries, upper_boundaries, 
			distribution_init, mean_init, sd_init, 120, 80, 500, 1e-7, processes)

		loglikelihood, Ne_estimate_BGS, u_estimate, s_mean_estimate, s_shape_estimate = result

		nlopt_function = partial(BinfL.optimisation_1N_classicBGS, mu = mu, observed_SFS_windows = SFS_array, 
			verbose = True, r_distance_array = r_distance_array, exon_array = exon_array, r_array = r_array, 
			mode = "pi", ploidy = ploidy, genomes = genomes, observed_Ne = single_Ne_estimate, round_B = False, 
			alpha = None, window_size = window_size)

		opt = nlopt.opt(nlopt.LN_NELDERMEAD, 4)
		opt.set_maxeval(400)
		opt.set_upper_bounds(list(np.array(upper_boundaries)*1.00001))
		opt.set_lower_bounds(lower_boundaries)
		opt.set_max_objective(nlopt_function)
		xopt = opt.optimize([Ne_estimate_BGS, u_estimate, s_mean_estimate, s_shape_estimate])
		Ne_estimate_BGS, u_estimate, s_mean_estimate, s_shape_estimate = xopt
		loglikelihood = opt.last_optimum_value()

		print("Ne, u, s_mean, s_shape: {}, {}, {}, {}".format(Ne_estimate_BGS, u_estimate, s_mean_estimate, s_shape_estimate))
		print("Loglikelihood: {}".format(loglikelihood, flush=True))


	if args["--selfing"] and ploidy == 2:

		print("Optimising 1N-classic-BGS SFS model with selfing...", flush=True)

		F = alpha / (2 - alpha)
		# use F to get some better initial guesses
		u_estimate = u_estimate / (1 + F)
		s_mean_estimate = s_mean_estimate / (1 + F)
		alpha_estimate = alpha

		print("[=] Conditioning on alpha = {}".format(alpha_estimate))

		upper_boundaries = [single_Ne_estimate * (1 + F) * 20, 2e-8, 0.25, 10]
		lower_boundaries = [single_Ne_estimate * (1 + F), 0, min_mean_s, 0.1]

		# how to deal with eps?

		nlopt_function = partial(BinfL.optimisation_1N_classicBGS_selfing, grad=None, mu = mu, observed_SFS_windows = SFS_array, 
			verbose = True, r_distance_array = r_distance_array, exon_array = exon_array, r_array = r_array, 
			mode = "pi", ploidy = ploidy, genomes = genomes, observed_Ne = single_Ne_estimate, 
			maximize = True, precision = 2, alpha = alpha_estimate, mask_hf = mask_hf, eps = f_eps, window_size = window_size)

		distribution_init = np.array(["u", "u", "lu", "lu"])
		mean_init = np.array([(lower_boundaries[0] + upper_boundaries[0]) / 2, 5e-9, (min_mean_s + 0.01), 1])
		sd_init = np.multiply(mean_init, np.array([3, 2, 2, 2.5]))

		result = BinfC.CRS2_LM(nlopt_function, lower_boundaries, upper_boundaries, 
			distribution_init, mean_init, sd_init, 120, 80, 800, 1e-7, processes)

		loglikelihood, Ne0_estimate_BGS, u_estimate, s_mean_estimate, s_shape_estimate = result

		minimize_boundaries = []

		for i in range(len(lower_boundaries)):
			minimize_boundaries.append((lower_boundaries[i], upper_boundaries[i]))

		minimize_function = partial(BinfL.optimisation_1N_classicBGS_selfing, grad=None, mu = mu, observed_SFS_windows = SFS_array, 
			verbose = True, r_distance_array = r_distance_array, exon_array = exon_array, r_array = r_array, 
			mode = "pi", ploidy = ploidy, genomes = genomes, observed_Ne = single_Ne_estimate, 
			maximize = False, precision = 2, alpha = alpha_estimate, mask_hf = mask_hf, eps = f_eps, window_size = window_size)

		result = minimize(fun=minimize_function, x0=np.array([Ne0_estimate_BGS, u_estimate, s_mean_estimate, s_shape_estimate]), 
			method="Nelder-Mead", bounds=minimize_boundaries, options={"maxfev" : 700, "adaptive" : 1})
			
		Ne0_estimate_BGS, u_estimate, s_mean_estimate, s_shape_estimate = result.x
		loglikelihood = -1 * result.fun

		print("Ne, u, s_mean, s_shape: {}, {}, {}, {}".format(Ne0_estimate_BGS, u_estimate, s_mean_estimate, s_shape_estimate))
		print("Loglikelihood: {}".format(loglikelihood, flush=True))

	else:

		print("Optimising 3N-classic-BGS SFS model...", flush=True)

		upper_boundaries = [Ne_estimate_BGS * 20, Ne_estimate_BGS * 20, Ne_estimate_BGS * 20, 
		single_Ne_estimate * 10, single_Ne_estimate * 10, 2e-8, 0.1, 10]
		lower_boundaries = [Ne_estimate_BGS / 20, Ne_estimate_BGS / 20, Ne_estimate_BGS / 20, 
		single_Ne_estimate / 10, single_Ne_estimate / 10, 0, min_mean_s, 0.1]

		nlopt_function = partial(BinfL.optimisation_3N_classicBGS, grad=None, mu = mu, observed_SFS_windows = SFS_array, 
			verbose = True, r_distance_array = r_distance_array, exon_array = exon_array, r_array = r_array, 
			mode = "SFS", ploidy = ploidy, genomes = genomes, observed_Ne = single_Ne_estimate, 
			maximize = True, precision = 0, alpha = None, mask_hf = mask_hf, eps = eps, window_size = window_size)

		minimize_function = partial(BinfL.optimisation_3N_classicBGS_mp, grad=None, mu = mu, observed_SFS_windows = SFS_array, 
			verbose = True, r_distance_array = r_distance_array, exon_array = exon_array, r_array = r_array, 
			mode = "SFS", ploidy = ploidy, genomes = genomes, observed_Ne = single_Ne_estimate, 
			maximize = False, precision = 2, alpha = None, mask_hf = mask_hf, eps = eps, processes = processes, 
			time_test = False, window_size = window_size)

		approx_B = single_Ne_estimate / Ne_estimate_BGS
		prior_Ne0 = Ne0_estimate / approx_B
		prior_Ne1 = Ne1_estimate / approx_B
		prior_Ne2 = Ne2_estimate / approx_B

		minimize_boundaries = []

		distribution_init = np.array(["n", "n", "n", "n", "n", "n", "ln", "ln"])
		mean_init = np.array([prior_Ne0, prior_Ne1, prior_Ne2, step0_estimate, step1_estimate, 
			u_estimate, s_mean_estimate, s_shape_estimate])
		sd_init = np.multiply(mean_init, np.array([1.5, 1.5, 1.5, 1, 1, 2, 2, 3]))

		result = BinfC.CRS2_LM(nlopt_function, lower_boundaries, upper_boundaries, 
			distribution_init, mean_init, sd_init, 300, 180, 1300, 1e-7, processes)

		loglikelihood, Ne0_estimate_BGS, Ne1_estimate_BGS, Ne2_estimate_BGS, step0_estimate_BGS, step1_estimate_BGS, u_estimate, s_mean_estimate, s_shape_estimate = result

		for i in range(len(lower_boundaries)):
			minimize_boundaries.append((lower_boundaries[i], upper_boundaries[i]))

		result = minimize(fun=minimize_function, x0=np.array([Ne0_estimate_BGS, Ne1_estimate_BGS, Ne2_estimate_BGS, 
		step0_estimate_BGS, step1_estimate_BGS, u_estimate, s_mean_estimate, s_shape_estimate]), 
		method="Nelder-Mead", bounds=minimize_boundaries, options={"maxfev" : 2500, "adaptive" : 1})

		Ne0_estimate_BGS, Ne1_estimate_BGS, Ne2_estimate_BGS, step0_estimate_BGS, step1_estimate_BGS, u_estimate, s_mean_estimate, s_shape_estimate = result.x
		loglikelihood = -1 * result.fun


		print("Ne0, Ne1, Ne2, step0, step1, u, s_mean, s_shape: {}, {}, {}, {}, {}, {}, {}, {}".format(Ne0_estimate_BGS, 
			Ne1_estimate_BGS, Ne2_estimate_BGS, step0_estimate_BGS, step1_estimate_BGS, 
			u_estimate, s_mean_estimate, s_shape_estimate))
		print("Loglikelihood: {}".format(loglikelihood, flush=True))

	sys.exit()
