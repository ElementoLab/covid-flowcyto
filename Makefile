.DEFAULT_GOAL := analysis


requirements:
	$(info    Installing python requirements)
	python -m pip install -r requirements.txt


data/matrix.pq:
	$(info    Parsing original data into metadata and matrix data)
	python -u src/parse_data.py


results/clinical/__done__: data/matrix.pq
	$(info    Running clinical analysis)
	python -u src/clinical.py


data/matrix_imputed.pq: data/matrix.pq
	$(info    Imputting any missing data)
	python -u src/imputation.py


results/unsupervised/__done__: data/matrix_imputed.pq
	$(info    Running unsupervised analysis)
	python -u src/unsupervised.py

results/supervised/__done__: data/matrix_imputed.pq
	$(info    Running supervised analysis)
	python -u src/supervised.py


data/fcs/__done__:
	python -u src/get_fcs.py


metadata/facs_dates.reduced.csv:
	$(info    Parsing batch date out of FCS files)
	python -u src/parse_batch.py


results/single_cell/__done__: data/fcs/__done__
	$(info    Running single cell analysis)
	python -u src/single_cell.py


analysis: \
	results/clinical/__done__ \
	results/unsupervised/__done__ \
	results/supervised/__done__ \
	results/single_cell/__done__


all: requirements analysis
	$(info    Finished with target all)


.PHONY: requirements analysis all
