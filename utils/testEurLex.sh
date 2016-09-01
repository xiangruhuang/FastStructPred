datafile=../../data/multilabel/Eur-Lex-Small.test
modelfile=../../model/multilabel/EurLex.model
testfile=EurLex.subtest
cp ../sparseLP/predict ./
for i in `seq $1 $2`; do
	head -$i ${datafile} | tail -1 > ${testfile}
	echo "============================================================================"
	./multilabel2uai -p multilabel -o ${testfile}.uai ${testfile} ${modelfile}
	#echo "============================================================================"
	#./predict -p uai -s 2 -o 0.01 -e 1 -m 100000 ${testfile}.uai ${modelfile}
	echo "============================================================================"
	rm AD3.log
	make AD3.log
	cat AD3.log | grep 'Best primal obj' | sed 's/.*Best primal obj = //' | sed 's/ sec.//' | sed 's/[\t| ].*[\t| ]/ /' | awk '{print $2 " " $1}' > log$i	
	rm AD3.out
done
