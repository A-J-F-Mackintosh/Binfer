#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Usage: Binfer_prep.py -v <STR> -b <STR> -g <STR> -d <INT> -D <INT> -w <INT> (-1 <STR> | -o STR>) (-2 <STR> | -t <STR) [-h]

  [Options]
    -v, --vcf <STR>                             VCF file
    -b, --bed <STR>                             Bed file of callable sites
    -g, --genomefile <STR>                      A genome with the lengths of chromosomes to sample
    -1, --taxon_1_samples <STR>                 A range of 1-based indices for samples in the #CHROM line, e.g. 1-16
    -o, --taxon_1_samples_alt <STR>             Indicies for samples, e.g. 1,4,6,7,8
    -2, --taxon_2_samples <STR>                 A range of 1-based indices for samples in the #CHROM line, e.g. 17-56
    -t, --taxon_2_samples_alt <STR>             Indicies for samples, e.g. 1,4,6,7,8
    -d, --downsample1 <INT>                    How many haploid genomes to downsample the SFS to (must be even)
    -D, --downsample2 <INT>                    How many haploid genomes to downsample the SFS to (must be even)
    -w, --window_size <INT>                     Number of bases in a window
    -h, --help                                  Show this message

"""

import sys
from docopt import docopt
import tskit
import pyslim
import msprime
import numpy as np
import math
import random
import warnings

sys.setrecursionlimit(10_000)

def n_choose_k(n, k):
	if k > n:
		return 0
	return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

# N = total haploid sample size
# K = derived allele frequency
# k = sampled derived alleles
# n = downsampled haploid sample size
def hypergeometric_dist(N, K, n, k):
	if N-K < n-k:
		return 0
	return 1.0 / (n_choose_k(N, n) // (n_choose_k(K, k) * n_choose_k(N-K, n-k)))

def multivariate_hypergeometric_dist(N, n, K1, k1, K2, k2, K3, k3):
	#print(N, n, K1, k1, K2, k2)
	return 1.0 * n_choose_k(K1, k1) * n_choose_k(K2, k2) * n_choose_k(K3, k3) / n_choose_k(N, n)

def find_window(start, idx, window_size):
	window_start = idx * window_size
	window_end = window_start + window_size
	if window_start <= start < window_end:
		return idx
	else:
		return find_window(start, idx+1, window_size)

def SFS_to_pi(SFS, genomes):
	n = genomes
	het = 0
	for i in range(0, len(SFS)):
		if SFS[i] > 0:
			het += SFS[i] * i*(n - i) / (n * (n - 1) * 0.5)
	return het / np.sum(SFS)


if __name__ == '__main__':
	__version__ = '0.1'
	args = docopt(__doc__)

	vcf_f = args["--vcf"]
	bed_f = args["--bed"]
	genome_f = args["--genomefile"]
	taxon_1_samples = args["--taxon_1_samples"]
	taxon_2_samples = args["--taxon_2_samples"]

	if args["--taxon_1_samples_alt"]:
		sample_idxs = args["--taxon_1_samples_alt"].split(",")
		taxon_1_sample_idxs = [int(i) for i in sample_idxs]
	else:
		taxon_1_sample_idxs = [int(i) for i in range(int(taxon_1_samples.split("-")[0]), int(taxon_1_samples.split("-")[1])+1)]

	if args["--taxon_2_samples_alt"]:
		sample_idxs = args["--taxon_2_samples_alt"].split(",")
		taxon_2_sample_idxs = [int(i) for i in sample_idxs]
	else:
		taxon_2_sample_idxs = [int(i) for i in range(int(taxon_2_samples.split("-")[0]), int(taxon_2_samples.split("-")[1])+1)]

	downsample_1 = int(args["--downsample1"])
	downsample_2 = int(args["--downsample2"])
	window_size = int(args["--window_size"])

	target_sequences = []
	windows = 0
	seq_to_pos = {} # gives number of nucleotide needed to add to a vcf pos to give its array pos

	print("[+] Reading genome files...")

	with open(genome_f, "r") as file:
		for line in file:
			target_seq, target_length = line.split("\t")[0:2]
			target_sequences.append(target_seq)
			seq_to_pos[target_seq] = windows * window_size
			target_windows = int(math.ceil(float(target_length) / window_size))
			windows += target_windows

	taxon_1_array = np.zeros((windows, downsample_1), dtype=float)
	taxon_2_array = np.zeros((windows, downsample_2), dtype=float)

	print("[=] Read {} sequences and made {} windows".format(len(target_sequences), windows))

	print("[+] Reading bed file...", flush=True)

	intervals = 0
	window_idx = 0

	with open(bed_f, "r") as file:
		for line in file:
			intervals += 1
			seq, start, end = line.split("\t")[0:3]

			if seq in target_sequences:

				start = int(start) + seq_to_pos[seq]
				end = int(end) + seq_to_pos[seq]

				window_idx = find_window(start, window_idx, window_size)

				contrib_to_window = min(((window_idx + 1) * window_size), end) - start
				taxon_1_array[window_idx][0] += contrib_to_window
				taxon_2_array[window_idx][0] += contrib_to_window

				if end > ((window_idx + 1) * window_size):
					looking_for_end = True

					while looking_for_end:
						window_idx += 1
						if end <= ((window_idx + 1) * window_size):
							contrib_to_window = end - (window_idx * window_size) 
							taxon_1_array[window_idx][0] += contrib_to_window
							taxon_2_array[window_idx][0] += contrib_to_window
							looking_for_end = False
						else:
							taxon_1_array[window_idx][0] += window_size
							taxon_2_array[window_idx][0] += window_size

	# think this is correct but need to test carefully with print statements

	print("[=] Read {} intervals".format(intervals), flush=True)

	print("[+] Reading VCF file...", flush=True)
	
	with open(vcf_f, "r") as file:
		for line in file:
			line = line.strip('\n')
			if line.startswith("#"):
				pass
			else:
				line_list = line.split("\t")
				sequence, pos = line_list[0:2]

				if sequence in target_sequences:

					idx = math.floor((int(pos) - 1 + seq_to_pos[sequence]) / window_size)

					alleles_taxon_1 = []
					alleles_taxon_2 = []
					genotypes_taxon_1 = []
					genotypes_taxon_2 = []

					# collect genotypes

					for i in range(9, len(line_list)):
							genotype = line_list[i].split(":")[0].replace('|', '/')
							if genotype == "./." or genotype == ".":
								pass
							else:
								if i-8 in taxon_1_sample_idxs:
									alleles_taxon_1 += genotype.split("/")
									genotypes_taxon_1.append(genotype)
								elif i-8 in taxon_2_sample_idxs:
									alleles_taxon_2 += genotype.split("/")
									genotypes_taxon_2.append(genotype)

					if len(alleles_taxon_1) >= downsample_1 and len(alleles_taxon_2) >= downsample_2: # enough genomes?

						if len(set(alleles_taxon_1)) + len(set(alleles_taxon_2)) == 3 and \
						len(set(alleles_taxon_1).intersection(set(alleles_taxon_2))) == 1: # check if biallelic and polarisable

							if len(set(alleles_taxon_1)) == 2: ## taxon2 is outgroup

								#print(genotypes_taxon_1)

								derived_allele = list(set(alleles_taxon_1) - set(alleles_taxon_2))[0]
								ancestral_allele = list(set(alleles_taxon_1) & set(alleles_taxon_2))[0]

								#print(derived_allele)
								#print(ancestral_allele)

								hom_derived_genotype = derived_allele + "/" + derived_allele
								hom_ancestral_genotype = ancestral_allele + "/" + ancestral_allele

								#print(hom_derived_genotype)
								#print(hom_ancestral_genotype)

								hom_derived_genotype_count = genotypes_taxon_1.count(hom_derived_genotype)
								hom_ancestral_genotype_count = genotypes_taxon_1.count(hom_ancestral_genotype)
								het_derived_genotype_count = len(genotypes_taxon_1) - hom_derived_genotype_count - hom_ancestral_genotype_count

								#print(hom_derived_genotype_count)
								#print(het_derived_genotype_count)

								N = len(genotypes_taxon_1)
								n = downsample_1 // 2

								#print(N, n)

								# how many homozygotes

								for k1 in range(0, min(n+1, hom_derived_genotype_count+1)):

									for k2 in range(0, min(n+1-k1, het_derived_genotype_count+1)):


										k3 = n - k1 - k2
										k = k1*2 + k2
										if k == n*2:
											k = 0

										taxon_1_array[idx][k] += multivariate_hypergeometric_dist(N, n, hom_derived_genotype_count, k1, het_derived_genotype_count, k2, hom_ancestral_genotype_count, k3)

								taxon_1_array[idx][0] += -1


							else:

								derived_allele = list(set(alleles_taxon_2) - set(alleles_taxon_1))[0]
								ancestral_allele = list(set(alleles_taxon_2) & set(alleles_taxon_1))[0]

								hom_derived_genotype = derived_allele + "/" + derived_allele
								hom_ancestral_genotype = ancestral_allele + "/" + ancestral_allele

								hom_derived_genotype_count = genotypes_taxon_2.count(hom_derived_genotype)
								hom_ancestral_genotype_count = genotypes_taxon_2.count(hom_ancestral_genotype)
								het_derived_genotype_count = len(genotypes_taxon_2) - hom_derived_genotype_count - hom_ancestral_genotype_count

								N = len(genotypes_taxon_2)
								n = downsample_2 // 2

								# how many homozygotes
								for k1 in range(0, min(n+1, hom_derived_genotype_count+1)):

									for k2 in range(0, min(n+1-k1, het_derived_genotype_count+1)):

										k3 = n - k1 - k2
										k = k1*2 + k2
										if k == n*2:
											k = 0

										taxon_2_array[idx][k] += multivariate_hypergeometric_dist(N, n, hom_derived_genotype_count, k1, het_derived_genotype_count, k2, hom_ancestral_genotype_count, k3)

								taxon_2_array[idx][0] += -1

						else:

							taxon_1_array[idx][0] += -1
							taxon_2_array[idx][0] += -1

					else:

						taxon_1_array[idx][0] += -1
						taxon_2_array[idx][0] += -1


	print("[=] Read SNPs")

	taxon_1_sfs = np.sum(taxon_1_array, axis=0)
	taxon_2_sfs = np.sum(taxon_2_array, axis=0)

	taxon_1_pi_estimate = SFS_to_pi(taxon_1_sfs, downsample_1)
	taxon_2_pi_estimate = SFS_to_pi(taxon_2_sfs, downsample_2)

	print("[=] Nucleotide diversity for taxon 1 : {}".format(taxon_1_pi_estimate))
	print("[=] Nucleotide diversity for taxon 2 : {}".format(taxon_2_pi_estimate))
	

	np.savetxt("taxon_1_full_array.txt", taxon_1_array, delimiter=',')
	np.savetxt("taxon_2_full_array.txt", taxon_2_array, delimiter=',')

	np.savetxt("taxon_1_sfs_array.txt", taxon_1_sfs, delimiter=',')
	np.savetxt("taxon_2_sfs_array.txt", taxon_2_sfs, delimiter=',')

	print("[=] Spectra written to file. Finished.")





