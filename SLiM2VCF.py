#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Usage: SLiM2VCF.py -p <STR> -P <INT> -n <INT> -N <INT> -s <INT> [-h -g -f]

  [Options]
    -p, --prefix <STR>                          Prefix of .trees .vcf and .sfs files
    -P, --ploidy <INT>                          Ploidy, 1 for haploid and 2 for diploid
    -n, --individuals <INT>                     Number of individuals to sample
    -N, --population <INT>                      Number of individuals in the population (haploid or diploid)
    -g, --genicfreq                             Write allele frequencies for genic regions
    -f, --Fis                                   Write Fis
    -s, --seed <INT>                            Random seed
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

warnings.simplefilter('ignore', msprime.TimeUnitsMismatchWarning)

def sample_ts(sample): # turn individual samples into diploid genomes
	new_sample = []
	for i in sample:
		new_sample.append(2*i)
		new_sample.append((2*i)+1)
	return new_sample

if __name__ == '__main__':
	__version__ = '0.1'
	args = docopt(__doc__)

	# deal with args
	ind = int(args["--individuals"])
	ploidy = int(args["--ploidy"])
	pop_size = int(args["--population"])
	vcf_f = args["--prefix"] +  "." + str(ploidy) + "." + str(ind) + ".vcf"
	sfs_f = args["--prefix"] + "." + str(ploidy) + "." + str(ind) + ".sfs"
	gaf_f = args["--prefix"] + "." + str(ploidy) + "." + str(ind) + ".gaf"
	fis_f = args["--prefix"] + "." + str(ploidy) + "." + str(ind) + ".fis"
	trees_f = args["--prefix"] + ".trees"
	
	r_seed = int(args["--seed"])
	random.seed(r_seed)

	ts = tskit.load(trees_f)


	# sample from the population
	if ploidy == 2:
		sample = sample_ts(random.sample([i for i in range(0, pop_size)], ind))
	elif ploidy == 1:
		sample = random.sample([i for i in range(0, pop_size)], ind)
	sts = ts.simplify(sample)


	# recapitate trees, if needed

	coal = 0
	no_coal = 0

	print("Overall Ne estimate = {}".format(sts.diversity(mode="branch") / 4))

	for tree in ts.trees():
		if len(tree.roots) > 1:
			no_coal += 1
		else:
			coal += 1

	print("Genealogies fully coalesced = {}".format(coal / (coal + no_coal)))

	if no_coal > 0: # recapitulate if some lineages fail to coalesce
		rts = pyslim.recapitate(ts, ancestral_Ne=(sts.diversity(mode="branch") / 4), recombination_rate=2.5e-8, random_seed=r_seed)
		srts = rts.simplify(sample)

	else:
		rts = ts
		srts = sts

	print("Overall Ne estimate (after recapitulation) = {}".format(srts.diversity(mode="branch") / 4))

	# get frequencies of deleterious mutations, if asked for
	if args["--genicfreq"]:

		genic_allele_freqs = rts.allele_frequency_spectrum(mode="site", polarised=True, span_normalise=False)
		np.savetxt(gaf_f, genic_allele_freqs, delimiter=',')

	# remove genic variation as we will only sample non-CDS regions
	srts = srts.delete_sites([site for site in range(0, len(srts.tables.sites))], record_provenance=True)

	m_ts = msprime.sim_mutations(srts, rate=7.5e-9, random_seed=r_seed) # add intergenic (and genic) variation

	#print(m_ts.sequence_length)

	windows = np.arange(0, m_ts.sequence_length, step=10_000)	# make window
	windows = np.append(windows, np.array([m_ts.sequence_length])) # make sure to include chrom end
	w_sfs = m_ts.allele_frequency_spectrum(windows=windows, mode="site", polarised=True, span_normalise=False) # get window sfs
	w_sfs[:, 0] = 10_000 - np.sum(w_sfs, 1) # make 0-tons monomorphic counts
	w_sfs = w_sfs[:, 0:ind*ploidy]

	#print(np.sum(w_sfs))

	windows = len(w_sfs)
	window_size = 10_000

	# we do not want to sample variation in CDS regions that was added by msprime
	# we have to do a (slow) loop over the CDS bed file to remove this variation from the spectra
	# there is surely a quicker way to do this (with less code too) but I have tested this and it works
	for chrom in [1, 2, 3]:
		with open("Caprub.chromosome" + str(chrom) + ".exons.bed", "r") as exon_file:
			for line in exon_file:
				if line.startswith("s"):
					pass
				else:
					seq, start, end = line.split("\t")
					if chrom == 1:
						start = int(start)
						end = int(end)
					elif chrom == 2:
						start = int(start) + 19570000
						end = int(end) + 19570000
					elif chrom == 3:
						start = int(start) + 33640000
						end = int(end) + 33640000
					init_window = int(math.floor(start / window_size))
					if init_window < windows:
						if end / window_size <= init_window + 1:
							exon_sfs = m_ts.allele_frequency_spectrum(windows=np.array([0, start, end, m_ts.sequence_length]), mode="site", 
								polarised=True, span_normalise=False)[1]
							exon_sfs[0] = end - start - np.sum(exon_sfs) # make 0-tons monomorphic counts
							exon_sfs = exon_sfs[0:ind*ploidy]
							w_sfs[init_window] = np.subtract(w_sfs[init_window], exon_sfs)

						else:
							exon_sfs = m_ts.allele_frequency_spectrum(windows=np.array([0, start, round((init_window + 1) * window_size), 
								m_ts.sequence_length]), mode="site", polarised=True, span_normalise=False)[1]
							exon_sfs[0] = round((init_window + 1) * window_size) - start - np.sum(exon_sfs) # make 0-tons monomorphic counts
							exon_sfs = exon_sfs[0:ind*ploidy]
							w_sfs[init_window] = np.subtract(w_sfs[init_window], exon_sfs)

							term_window = int(math.floor(end / window_size))
							if term_window < windows:
								exon_sfs = m_ts.allele_frequency_spectrum(windows=np.array([0, round(term_window * window_size), 
									end, m_ts.sequence_length]), mode="site", polarised=True, span_normalise=False)[1]
								exon_sfs[0] = end - round(term_window * window_size) - np.sum(exon_sfs) # make 0-tons monomorphic counts
								exon_sfs = exon_sfs[0:ind*ploidy]
								w_sfs[term_window] = np.subtract(w_sfs[term_window], exon_sfs)

							for i in range(init_window + 1, term_window):
								w_sfs[i] = w_sfs[i] * 0

	#print(np.sum(w_sfs))

	np.savetxt(sfs_f, w_sfs, delimiter=',')

	with open(vcf_f, "w") as file:
		m_ts.write_vcf(file, contig_id="0")

	# if Fis, then calculate this from the vcf just written to file

	if args["--Fis"]:

		#H_e_list = []
		#H_o_list = []
		b_list = []
		c_list = []

		with open(vcf_f, "r") as file:
			for line in file:
				line = line.strip('\n')
				if line.startswith("#"):
					pass
				else:
					genotypes = [line.split("\t")[i] for i in range(9, 9+ind)]
					#print(genotypes)
					H_o = 0
					H_o += genotypes.count("0|1")
					H_o += genotypes.count("1|0")
					H_o = H_o / ind
					alleles = [int(allele) for genotype in genotypes for allele in genotype.split("|")]
					#print(alleles)
					if 2 in alleles:
						pass
					else:
						p = np.sum(alleles) / (2 * ind)
						b = ind / (ind - 1) * (p * (1 - p) - (2 * ind - 1) / (4 * ind) * H_o)
						c = H_o / 2
						b_list.append(b)
						c_list.append(c)
						
		overall_c = np.sum(c_list) / len(c_list)
		overall_b = np.sum(b_list) / len(b_list)
		overall_Fis = 1 - (overall_c / (overall_b + overall_c))
		
		np.savetxt(fis_f, np.array([overall_Fis]), delimiter=',')

