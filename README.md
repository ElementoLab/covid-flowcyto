# COVID19 profiling of peripheral immune system with flow cytometry


## Organization

- [This CSV file](data/original/clinical_data.joint.20200723.csv) contains the original manually curated data
- The [metadata](metadata) directory contains parsed/further curated metadata
- Raw data (e.g. CSV or FCS files) will be under the [data](data) directory.
    - FCS files can be get from Cytobank (requires account, described below).
    - H5ad files [can be get from the following URLs](metadata/h5ad_urls.json) using the `make get_h5ad` command (see more below).
- The [src](src) directory contains source code used to analyze the data
- A [Makefile](Makefile) is provided to allow easy execution of task blocks.
- Outputs from the analysis will be present in a `results` directory, with subfolders pertaining to each part of the analysis described below.


FCS files are hosted on Cytobank. An account is needed to download the files, which can be made programmatically with the `make get_fcs` command (see below).
To connect with your account, simply add your credentials to a file named `~/.cytobank.auth.json` containing the fields `username` and `password`:
```json
{"username": "username", "password": "ABCD1234"}
```
Be sure to make the file read-only (e.g. `chmod 400 ~/.cytobank.auth.json`).

## Reproducibility

### Running

To see all available steps type:
```bash
make help
```
```
Makefile for the covid-flowcyto package/project.
Available commands:
help            Display help and quit
requirements    Install software requirements with pip
get_fcs         Download FCS files from Cytobank
get_h5ad        Download H5ad files
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
make requirements
make
```

### Requirements

- Python 3.7+ (was run on 3.8.2)
- Python packages as specified in the [requirements file](requirements.txt) - install with `make requirements` or `pip install -r requirements.txt`.


### Virtual environment

It is recommended to compartimentalize the analysis software from the system's using virtual environments, for example.

Here's how to create one with the repository and installed requirements:

```bash
git clone git@github.com:ElementoLab/covid-flowcyto.git
cd covid-flowcyto
virtualenv .venv
source activate .venv/bin/activate
pip install -r requirements.txt
```

### Raw data
