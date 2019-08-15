src_dir=exg_det


for main_dir in ../system_testing_results/*/; do
	
	echo "DIR: ${main_dir}"
	
	for dir in ${main_dir}/*/; do
		dir=${dir%*/}
		dir=${dir##*/}
		echo "DIR: ${main_dir}${dir}"
		python evaluation_runner.py ${main_dir}/${dir}/per_row_results/ ${main_dir}${dir}/

	done
	echo "================================================"
done
