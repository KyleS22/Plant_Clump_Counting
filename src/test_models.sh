
python validation_runner.py conventional_ML/final_code_and_models/models/SVM_GLCM_model.sav ~/Plant_Counting_Data/exg_det ../system_testing_results/SVM_GLCM "SVM_GLCM.csv" "GLCM" --real_counts_path ~/Plant_Counting_Data/real_counts.csv  --sys_test


python validation_runner.py conventional_ML/final_code_and_models/models/KNN_GLCM_model.sav ~/Plant_Counting_Data/exg_det ../system_testing_results/KNN_GLCM "KNN_GLCM.csv" "GLCM" --real_counts_path ~/Plant_Counting_Data/real_counts.csv  --sys_test

python validation_runner.py conventional_ML/final_code_and_models/models/GNB_GLCM_model.sav ~/Plant_Counting_Data/exg_det ../system_testing_results/GNB_GLCM "GNB_GLCM.csv" "GLCM" --real_counts_path ~/Plant_Counting_Data/real_counts.csv  --sys_test


python validation_runner.py conventional_ML/final_code_and_models/models/KNN_FFT_model.sav ~/Plant_Counting_Data/exg_det ../system_testing_results/KNN_FFT "KNN_FFT.csv" "FFT" --real_counts_path ~/Plant_Counting_Data/real_counts.csv  --sys_test


python validation_runner.py conventional_ML/final_code_and_models/models/KNN_LBPH_model.sav ~/Plant_Counting_Data/exg_det ../system_testing_results/KNN_LBPH "KNN_LBPH.csv" "LBPH" --real_counts_path ~/Plant_Counting_Data/real_counts.csv  --sys_test


python validation_runner.py conventional_ML/final_code_and_models/models/GNB_FFT_model.sav ~/Plant_Counting_Data/exg_det ../system_testing_results/GNB_FFT "GNB_FFT.csv" "FFT" --real_counts_path ~/Plant_Counting_Data/real_counts.csv  --sys_test


python validation_runner.py conventional_ML/final_code_and_models/models/GNB_LBPH_model.sav ~/Plant_Counting_Data/exg_det ../system_testing_results/GNB_LBPH "GNB_LBPH.csv" "LBPH" --real_counts_path ~/Plant_Counting_Data/real_counts.csv  --sys_test


python validation_runner.py conventional_ML/final_code_and_models/models/SVM_FFT_model.sav ~/Plant_Counting_Data/exg_det ../system_testing_results/SVM_FFT "SVM_FFT.csv" "FFT" --real_counts_path ~/Plant_Counting_Data/real_counts.csv  --sys_test


python validation_runner.py conventional_ML/final_code_and_models/models/SVM_LBPH_model.sav ~/Plant_Counting_Data/exg_det ../system_testing_results/SVM_LBPH "SVM_LBPH.csv" "LBPH" --real_counts_path ~/Plant_Counting_Data/real_counts.csv  --sys_test


python validation_runner.py ../trained_models/EncoderCNN_NoSynth/EncoderCNN_NoSynth.json ~/Plant_Counting_Data/exg_det ../system_testing_results/CNN "CNN.csv" "ENCODER" --path_to_weights ../trained_models/EncoderCNN_NoSynth/EncoderCNN_NoSynth.h5 --real_counts_path ~/Plant_Counting_Data/real_counts.csv  --sys_test


