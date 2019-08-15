
# Good
python validation_runner.py conventional_ML/final_code_and_models/models/SVM_GLCM_model.sav ~/Plant_Counting_Data/combined_val ../final_testing_results/ "SVM_GLCM.csv" "GLCM"


python validation_runner.py conventional_ML/final_code_and_models/models/KNN_GLCM_model.sav ~/Plant_Counting_Data/combined_val ../final_testing_results/ "KNN_GLCM.csv" "GLCM"

# Good
python validation_runner.py conventional_ML/final_code_and_models/models/GNB_GLCM_model.sav ~/Plant_Counting_Data/combined_val ../final_testing_results/ "GNB_GLCM.csv" "GLCM"


python validation_runner.py conventional_ML/final_code_and_models/models/KNN_FFT_model.sav ~/Plant_Counting_Data/combined_val ../final_testing_results/ "KNN_FFT.csv" "FFT"

python validation_runner.py conventional_ML/final_code_and_models/models/KNN_LBPH_model.sav ~/Plant_Counting_Data/combined_val ../final_testing_results/ "KNN_LBPH.csv" "LBPH"

# Good
python validation_runner.py conventional_ML/final_code_and_models/models/GNB_FFT_model.sav ~/Plant_Counting_Data/combined_val ../final_testing_results/ "GNB_FFT.csv" "FFT"

# Good
python validation_runner.py conventional_ML/final_code_and_models/models/GNB_LBPH_model.sav ~/Plant_Counting_Data/combined_val ../final_testing_results/ "GNB_LBPH.csv" "LBPH"

# Good
python validation_runner.py conventional_ML/final_code_and_models/models/SVM_FFT_model.sav ~/Plant_Counting_Data/combined_val ../final_testing_results/ "SVM_FFT.csv" "FFT"

# Good
python validation_runner.py conventional_ML/final_code_and_models/models/SVM_LBPH_model.sav ~/Plant_Counting_Data/combined_val ../final_testing_results/ "SVM_LBPH.csv" "LBPH"

# Good
python validation_runner.py ../trained_models/EncoderCNN_NoSynth/EncoderCNN_NoSynth.json ~/Plant_Counting_Data/combined_val ../final_testing_results/ "CNN.csv" "ENCODER" --path_to_weights ../trained_models/EncoderCNN_NoSynth/EncoderCNN_NoSynth.h5

mv ../final_testing_results/*conf_matrix* ../final_testing_results/conf_mats/
mv ../final_testing_results/*.csv ../final_testing_results/test_results/
