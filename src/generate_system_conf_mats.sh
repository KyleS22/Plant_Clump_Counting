src_dir=faster_det

for dir in ../system_testing_results/${src_dir}/*/; do
	dir=${dir%*/}
	dir=${dir##*/}
	python visualize_conf_matrix.py ../system_testing_results/${src_dir}/${dir}/${dir}_conf_matrix.csv --save ../system_testing_results/${src_dir}/${dir}/pretty_${dir}_conf_matrix.png --silent
done
