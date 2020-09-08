.DEFAULT_GOAL := analysis


NAME=$(shell basename `pwd`)
DATE=$(shell date '+%Y%m%d-%H%M%S')
$(shell mkdir -p log)

help:  ## Display help and quit
	@echo Makefile for the $(NAME) package/project.
	@echo Available commands:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		%s\n", $$1, $$2}'


requirements:  ## Install software requirements with pip
	$(info    Installing python requirements)
	python -m pip install -r requirements.txt \
		2>&1 | tee log/requirements.$(DATE).log


# Get data as FCS files
data/fcs/__done__:
	python -u \
		src/get_fcs.py \
			2>&1 | tee log/get_fcs.$(DATE).log
get_fcs: data/fcs/__done__   ## Download FCS files from Cytobank

metadata/facs_dates.reduced.csv:
	$(info    Parsing batch date out of FCS files)
	python -u \
		src/parse_batch.py \
			2>&1 | tee log/parse_batch.$(DATE).log
get_batch: metadata/facs_dates.reduced.csv  ## Parse processing date from FCS metadata


# Get data as h5ad files
data/h5ad/__done__:
	python -u \
		src/get_h5ad.py \
			2>&1 | tee log/get_h5ad.$(DATE).log
get_h5ad: data/h5ad/__done__   ## Download h5ad files

metadata/facs_dates.reduced.csv:
	$(info    Parsing batch date out of FCS files)
	python -u \
		src/parse_batch.py \
			2>&1 | tee log/parse_batch.$(DATE).log
get_batch: metadata/facs_dates.reduced.csv  ## Parse processing date from FCS metadata


# Metadata / data cleanup
data/matrix.pq:
	$(info    Parsing original data into metadata and matrix data)
	python -u \
		src/parse_data.py \
			2>&1 | tee log/parse_date.$(DATE).log
parse: data/matrix.pq  ## Parse original data into metadata and matrix data

data/matrix_imputed.pq: data/matrix.pq
	$(info    Imputting any missing data)
	python -u \
		src/imputation.py \
			2>&1 | tee log/imputation.$(DATE).log
impute: data/matrix_imputed.pq  ## Imputation of missing FACS data

# Analysis
results/clinical/__done__: data/matrix.pq
	$(info    Running clinical analysis)
	python -u \
		src/clinical.py \
			2>&1 | tee log/clinical.$(DATE).log
clinical: results/clinical/__done__  ## Run analysis of clinial data

results/unsupervised/__done__: data/matrix_imputed.pq
	$(info    Running unsupervised analysis)
	python -u \
		src/unsupervised.py \
			2>&1 | tee log/unsupervised.$(DATE).log
	python -u \
		src/unsupervised_pseudotime.py \
			2>&1 | tee log/unsupervised_pseudotime.$(DATE).log
unsupervised: results/unsupervised/__done__  ## Run unsupervised analysis
results/supervised/__done__: data/matrix_imputed.pq
	$(info    Running supervised analysis)
	python -u \
		src/supervised_fit_models.py \
			2>&1 | tee log/supervised_fit_models.$(DATE).log
	python -u \
		src/supervised_plot_results.py \
			2>&1 | tee log/supervised_plot_results.$(DATE).log
	python -u \
		src/supervised_plot_jointly.py \
			2>&1 | tee log/supervised_plot_jointly.$(DATE).log
supervised: results/supervised/__done__  ## Run supervised analysis
results/temporal/__done__: data/fcs/__done__
	$(info    Running single cell analysis)
	python -u \
		src/temporal.py \
			2>&1 | tee log/temporal.$(DATE).log
temporal: results/temporal/__done__  ## Run temporal analysis
results/single_cell/__done__: data/fcs/__done__
	$(info    Running single cell analysis)
	python -u \
		src/single_cell_prepare.py \
			2>&1 | tee log/single_cell_prepare.$(DATE).log
	python -u \
		src/single_cell_analysis.py \
			2>&1 | tee log/single_cell_analysis.$(DATE).log
single_cell: results/single_cell/__done__  ## Run single-cell analysis

analysis: \
	results/clinical/__done__ \
	results/unsupervised/__done__ \
	results/supervised/__done__ \
	results/temporal/__done__ \
	results/single_cell/__done__


all: requirements analysis  ## Run all analysis steps in order
	$(info    Finished with target all)


.PHONY: requirements analysis all
