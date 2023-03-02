#################@#################@#################@#################
#################@#################@#################@#################
J=jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
# JN=$(J) --to markdown  --stdout # for dev
JN=$(J) --to notebook  --inplace # for the final touch
#################@#################@#################@#################
#################@#################@#################@#################
all: 
	$(JN) UltraFastCat.ipynb

#################@#################@#################@#################
update:
	python3 -m pip install --upgrade -r requirements.txt

push_data:
	rsync -av  -e "ssh  -i ~/.ssh/id-ring-ecdsa" ../data laurent@10.164.7.21:metagit/JNJER/
#################@#################@#################@#################
