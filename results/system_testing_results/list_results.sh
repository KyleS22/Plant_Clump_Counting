
basedir=$1


for dir in ${basedir}/*; do
	
	echo "=============================="
	base=$(basename $dir)
	echo "${base} RESULTS"	
	cat ${dir}/model_results.csv
	
	echo " "
	echo "${base} MODE RESULTS"
	cat ${dir}/model_mode_results.csv
	echo "============================="
done	
