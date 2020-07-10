# COVID19 profiling of peripheral immune system with flow cytometry


## Organization

- [This CSV file](data/original/clinical_data.joint.20200710.csv) contains the original manually curated data
- The [metadata](metadata) directory contains parsed/further curated metadata
- The [src](src) directory contains source code used to analyze the data


## Reproducibility

To see all available steps type:
```bash
$ make help
```
```
Makefile for the covid-facs package/project.
Available commands:
help            Display help and quit
requirements    Install software requirements with pip
get_fcs         Download FCS files from Cytobank
get_batch       Parse processing date from FCS metadata
parse           Parse original data into metadata and matrix data
impute          Imputation of missing FACS data
clinical        Run analysis of clinial data
unsupervised    Run unsupervised analysis
supervised      Run supervised analysis
temporal        Run temporal analysis
single_cell     Run single-cell analysis
all             Run all analysis steps in order
```

To reproduce analysis, simply do:

```bash
$ make requirements
$ make
```
