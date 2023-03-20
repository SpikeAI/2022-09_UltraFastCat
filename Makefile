#################@#################@#################@#################
#################@#################@#################@#################
J=jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to markdown  --stdout # for dev
# JN=$(J) --to notebook  --inplace # for the final touch
#################@#################@#################@#################
#################@#################@#################@#################
all: 
	$(JN) UltraFastCat.ipynb

#################@#################@#################@#################
update:
	python3 -m pip install --upgrade -r requirements.txt

pull_babbage:
	rsync -av  -e "ssh  -i ~/.ssh/id-ring-ecdsa"  laurent@10.164.7.21:metagit/JNJER/2023-03-09_UltraFastCat/cached_data .

pull_darwin:
	rsync -av  -e "ssh  -i ~/.ssh/id-ring-ecdsa"  perrinet.l@10.164.6.190:metagit/JNJER/2023-03-09_UltraFastCat/cached_data .

HOST=laurent@10.164.7.21# babbage
HOST=perrinet.l@10.164.6.190# darwin
push_data:
	rsync -av  -e "ssh  -i ~/.ssh/id-ring-ecdsa" ../DataSetMaker $(HOST):metagit/JNJER/
	rsync -av  -e "ssh  -i ~/.ssh/id-ring-ecdsa" ../data $(HOST):metagit/JNJER/
#################@#################@#################@#################
