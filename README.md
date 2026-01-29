# Binfer
Code and example data for fitting models of background selection with demography and partial-selfing

### Installation

Dependencies are best addressed via conda.

```
conda create -n Binfer && conda activate Binfer && conda install dadi=2.3.3 docopt=0.6.2 moments=1.1.16 nlopt=2.7.1 numba=0.59.1 numpy=1.26.4 scipy=1.13.1

git clone https://github.com/A-J-F-Mackintosh/Binfer.git
```

### Usage

The input for fitting BGS models is unfolded windowed site frequency spectra. The script `Binfer_prep.py` can be used to generate this from a clean VCF and a bed file of callable sites.

#### Preparing data

The python script `Binfer_prep.py` can be used to prepare data for fitting BGS model.

```

Usage: Binfer_prep.py -v <STR> -b <STR> -g <STR> -d <INT> -D <INT> -w <INT> (-1 <STR> | -o STR>) (-2 <STR> | -t <STR) [-h -a <INT>]

  [Options]
    -v, --vcf <STR>                             VCF file
    -b, --bed <STR>                             Bed file of callable sites
    -g, --genomefile <STR>                      A genome with the lengths of chromosomes to sample
    -1, --taxon_1_samples <STR>                 A range of 1-based indices for samples in the #CHROM line, e.g. 1-16
    -o, --taxon_1_samples_alt <STR>             Indicies for samples, e.g. 1,4,6,7,8
    -2, --taxon_2_samples <STR>                 A range of 1-based indices for samples in the #CHROM line, e.g. 17-56
    -t, --taxon_2_samples_alt <STR>             Indicies for samples, e.g. 1,4,6,7,8
    -d, --downsample1 <INT>                     How many haploid genomes to downsample the SFS to (must be even)
    -D, --downsample2 <INT>                     How many haploid genomes to downsample the SFS to (must be even)
    -a, --use_ancestral_allele <INT>            Instead of using an outgroup, use ancestral allele field on VCF. If True, args for taxon_2 are ignored. [default: 0]
    -w, --window_size <INT>                     Number of bases in a window
    -h, --help                                  Show this message

```

If the VCF contains data from two different species which can be used to polarise SNPs, then an example `Binfer_prep` command would be:

`Binfer/Binfer_prep.py -v corientalis_grandiflora.SNPS.callable.vcf -b corientalis_grandiflora.callable.bed -g genomefile.txt -1 1-16 -2 17-66 -d 20 -D 20 -w 10_000 > Binfer_prep.log`

The above command generates arrays of unfolded site frequency spectra along the genome for taxon_1 and taxon_2. Note that taxon_2 could just be a single outgroup sample to polarise alleles in taxon_1, or it could be a sample of interest as above.

If the VCF only contains data from a single species and there is an ancestral allele field in the VCF, then an example `Binfer_prep` command would be:

```
Binfer/Binfer_prep.py -v drosophila_melanogaster.no_CDS.vcf -b drosophila_melanogaster.no_CDS.bed -g drosophila_melanogaster.genomefile -1 1-36 -2 1-36 -d 20 -D 20 -a 1 -w 10_000
```

The code does not currently support analysis of folded spectra, although this would be straightforward to implement. If you are interested in fitting BGS models to folded data then please open a github issue.

#### Fitting models

BGS models can be fit with `Binfer.py`.

```
Usage: Binfer.py -s <STR> -b <STR> -g <STR> -r <STR> -p <INT> -n <INT> -w <INT> -m <FLT> -e <INT> [-h -f -u -a -y <FLT> -q <INT> -Z <FLT> -P <STR> -F <FLT> -H]

  [Options]
    -s, --sfs <STR>                            SFS file
    -b, --bed <STR>                            Bed file of sites under purifying selection (e.g. CDS)
    -g, --genomefile <STR>                     Genomefile of sequence lengths
    -r, --rmaps <STR>                          File with paths to recombination maps
    -p, --ploidy <INT>                         Ploidy, 1 for haploid and 2 for diploid
    -n, --individuals <INT>                    Number of individuals in sample
    -w, --windowsize <INT>                     Bases in each window
    -m, --mu <FLT>                             De novo mutation rate per-site per-generation
    -f, --selfing                              Fit models with partial selfing rather than random mating
    -u, --fixu                                 Fix the deleterious mutation rate to the de novo rate (this assumes that intervals in bed are CDS, with 0.725 sites nonsyn and 0.275 syn)
    -a, --mask                                 Mask last two SFS entries which are prone to reference bias
    -y, --eps <FLT>                            Use a specific polarisation error rate
    -q, --processes <INT>                      Number of CPUs to use (default is 1)
    -Z, --low_B_mask <FLT>                     Windows in the lowest FLT of B values will be masked and not contribute to parameter estimates
    -e, --epochs <INT>                         Number of epochs for randomly mating models (2 or 3)
    -P, --prefix <STR>                         A prefix for writing files (default is Binfer)
    -F, --Fis <FLT>                            Use this value of Fis to calculate the selfing rate (alpha), rather than estimating alpha from the SFS
    -H, --Hapmap                               Recombination maps are in Hapmap format
    -h, --help                                 Show this message
```

Where an example command would look like:

`./Binfer.v1/Binfer.py -s grandiflora_full_array.txt -b corientalis_grandiflora.CDS.bed -g genomefile.txt -r rmap_files.txt -p 2 -n 10 -w 10_000 -m 7e-9 -a -q 2`

The one non-standard file type in the input is `rmap_files.txt`. This is text file with a list of file names, each corresponding to a recombination map. For example:

```
Capsella_rubella_Slotte2013_chromosome1.txt
Capsella_rubella_Slotte2013_chromosome2.txt
Capsella_rubella_Slotte2013_chromosome3.txt
```

The recombination map is assumed to be in **2-column bed format** with rates in **cM/Mb** (see below). Hapmap format can be used by setting `--Hapmap`.

A recombination map in full bed format:
```
0        10000    1.0
10000    20000    2.0
20000    30000    0.8
```
In 2-column bed format:
```
10000    1.0
20000    2.0
30000    0.8
```
In Hapmap format:
```
0        1.0
10000    2.0
20000    0.8
30000    0
```


