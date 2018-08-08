for i in $@
do
	python drs2tuple.py ${i}.drs > ${i}.tuple
	python d-match.py -f1 ${i}.tuple -f2 dev.tuple -pr -r 100 -p 25 > ${i}.eval.100
done
