src_dir=exg_det

for dir in ../system_testing_results/${src_dir}/*/; do
	dir=${dir%*/}
	dir=${dir##*/}
	python evaluation_runner.py ../system_testing_results/${src_dir}/${dir}/per_row_results/ ../system_testing_results/${src_dir}/${dir}/
done
