default: run

J=jupyter nbconvert  --ExecutePreprocessor.timeout=0 --allow-errors --execute
# J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

run:
	$(JN) UltraFastCat.ipynb
	git commit -m 'results notebook: full run' -a
	git push


pull_babbage:
	rsync -av -u  -e "ssh  -i ~/.ssh/id-ring-ecdsa"  laurent@10.164.7.21:metagit/JNJER/2022-03_UltraFastCat/* .
clean:
	rm -fr /tmp/2022-02-08_*
