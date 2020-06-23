.DEFAULT_GOAL := analysis

requirements:
	$(info    Installing python requirements)
	python -m pip install -r requirements.txt

data/matrix.pq:
	$(info    Parsing original data into metadata and matrix data)
	python -u src/parse_data.py

data/matrix_imputed.pq: data/matrix.pq
	$(info    Imputting any missing data)
	python -u src/imputation.py


results/unsupervised/__done__: data/matrix_imputed.pq
	$(info    Running unsupervised analysis)
	python -u src/unsupervised.py

results/supervised/__done__:
	$(info    Running supervised analysis)
	python -u src/supervised.py


analysis: results/unsupervised/__done__ results/supervised/__done__


# analysis_job: summarize mklog
# 	sbatch -p longq --time 8-00:00:00 -c 12 --mem 80000 \
# 	-J covid-immune.analysis \
# 	-o log/$(shell date +"%Y%m%d-%H%M%S").covid-immune.analysis.log \
# 	--x11 --wrap "ngs_analysis metadata/project_config.yaml"

all: requirements analysis
	$(info    Finished with target all)

.PHONY: requirements analysis all
