# Binfer
Scripts and example data for fitting models of background selection with demography and partial-selfing

### Usage

#### Preparing data

The python script `Binfer_prep.py` can be used to prepare data for fitting BGS model.

```
Usage: Binfer_prep.py -v <STR> -b <STR> -g <STR> -d <INT> -D <INT> -w <INT> (-1 <STR> | -o STR>) (-2 <STR> | -t <STR) [-h]

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
    -w, --window_size <INT>                     Number of bases in a window
    -h, --help                                  Show this message
```

Where an example command would look like:

`./Binfer_prep.py -v corientalis_grandiflora.SNPS.callable.vcf -b corientalis_grandiflora.callable.bed -g genomefile.txt -1 1-16 -2 17-66 -d 20 -D 20 -w 10_000 > Binfer_prep.log`

Note that the vcf file must already be filtered on callable intervals. In other words, the vcf should only include SNPs were there is good sequencing coverage across samples.

The above command generates arrays of unfolded spectra for taxon_1 and taxon_2. If taxon_2 is an outgroup that will not be analysed then the output for this taxon can be ignored.

#### Fitting models

BGS models can be fit with `Binfer.v1.py`.

```
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
```

Where an example command would look like:

`./Binfer.v1/Binfer.v1.py -s grandiflora_full_array.txt -b corientalis_grandiflora.CDS.bed -g genomefile.txt -r rmap_files.txt -p 2 -n 10 -w 10_000 -m 7e-9 -a -q 2`

The one non-standard file type in the input is `rmap_files.txt`. This is text file with a list of file names, each corresponding to a recombination map. For example:

```
Capsella_rubella_Slotte2013_chromosome1.txt
Capsella_rubella_Slotte2013_chromosome2.txt
Capsella_rubella_Slotte2013_chromosome3.txt
```

Where `Capsella_rubella_Slotte2013_chromosome1.txt` gives recombination rates across chromosome 1 as:

```
phys    rec
0.55    0.686264135918197
0.65    0.693009333427109
0.75    0.717790258983784
```

with `phys` being the end coordinate for a genomic interval (1-based, Mb) and `rec` being the recombination rate in that interval (cM/Mb).

Currently, if `Binfer.v1.py` is called without `-f` (partial-selfing awareness) then the following models are fit sequentially:

```
1-epoch neutral
2-epoch neutral
3-epoch neutral
Classic-BGS (1-epoch)
BGS-with-demography (3-epoch)
```

If `Binfer.v1.py` is called with `-f` then the following models are fit sequentially:

```
1-epoch neutral
2-epoch neutral
Classic-BGS (1-epoch)
BGS-with-demography and partial-selfing (1-epoch)
```


