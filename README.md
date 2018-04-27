# DotcodeR

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.839597.svg)](https://doi.org/10.5281/zenodo.839597)

**Alignment-free comparative genomic screen for structured RNAs using coarse-grained secondary structure dot plots**

Last updated: 2018-04-27

Structured non-coding RNAs play many different roles in the cells, but the annotation of these RNAs is lacking even within the human genome. The currently available computational tools are either too computationally heavy for use in full genomic screens or rely on pre-aligned sequences.  
Here we present a fast and efficient method DotcodeR for detecting structurally similar RNAs in genomic sequences by comparing their corresponding coarse-grained secondary structure dot plots at string level. This allows us to perform an all-versus-all scan of all window pairs from two genomes without alignment. Our computational experiments with simulated data and real chromosomes demonstrate that the presented method has good sensitivity. DotcodeR can be useful as a pre-filter in a genomic comparative scan for structured RNAs.

## Installation
* DotcodeR (ver. 1.0.1) (**dotcoder-1.0.1.tar.gz**) in C++ program

### Requirements
* ViennaRNA package (>= 2.0)

### Install on Linux and macOS
Type the followings in your terminal:

```
$ tar zxf dotcoder-1.0.1.tar.gz  
$ cd dotcoder-1.0.1  
$ ./configure --with-vienna-rna=/path/to/vienna-rna
```
Then,
```
$ make  
$ sudo make install
```

## Usage
```
$ dotcoder [options]* [fasta]+

[fasta]
 At most two (gzipped) FASTA files

[options]
 -D, --dna             Read DNA sequences for input
                        (default: off, i.e. RNA sequence)
 -z, --gzip            Allow gzipped files for input
                        (default: off)

 -d, --dist <INT>      Sliding window size of 2d+1 in the dot plots
                        (default <INT> = 1)
 -w, --winsize <INT>   Sliding window size of w in the genomic sequences
                        (default <INT> = 120)
 -s, --step <INT>      Step size of s of the sliding window
                        (default <INT> = 30)
 -c, --cutoff <INT>    Report only similarity scores >= cutoff
                        (default <INT> = 20)

 -v, --verbose         Show details
                        (default: off)
 -h, --help            Show this message
 -H, --full-help       Show detailed help
```

## Data
* Training set for benchmarking DotcodeR (**Training_set.tar.gz**)  
* Test set for benchmarking DotcodeR (**Test_set.tar.gz**)

## Reference
Yuki Kato, Jan Gorodkin and Jakob Hull Havgaard,
**Alignment-free comparative genomic screen for structured RNAs using coarse-grained secondary structure dot plots**,
*BMC Genomics*, vol. 18, 935, 2017.

---
If you have any questions, please contact [Yuki Kato](http://www.med.osaka-u.ac.jp/pub/rna/ykato/)  
*Graduate School of Medicine, Osaka University, Japan*  
*Center for non-coding RNA in Technology and Health (RTH), University of Copenhagen, Denmark*
