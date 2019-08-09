

for filename in ../final_testing_results/conf_mats/*.csv; do
	out_name=$(basename $filename)
	new_out_name=${out_name%.csv}.png
	python visualize_conf_matrix.py "$filename" --save ../final_testing_results/pretty_conf_mats/$new_out_name --silent
done
