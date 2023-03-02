DIR=2023-02-20_BiolCybernetics

J=jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to markdown  --stdout # for dev
# JN=$(J) --to notebook  --inplace # for the final touch

#################@#################@#################@#################
#################@#################@#################@#################
all: 
	$(JN) UltraFastCat.ipynb

#################@#################@#################@#################

#################@#################@#################@#################

update:
	python3 -m pip install --upgrade -r requirements.txt
