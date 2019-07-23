MODEL_NAME=Test_Model
SAVE_DIR=~/Plant_Clump_Counting/trained_models/Test_Model

MODEL_SAVE_NAME="$MODEL_NAME.h5"
MODEL_SAVE_PATH="$SAVE_DIR/$MODEL_SAVE_NAME"

TEST_NAME="${MODEL_NAME}_test_scores.csv"

python counting_CNN/train_runner.py ~/Plant_Counting_Data/sorted_synthetic/ --model_save_dir $SAVE_DIR --validation_data_dir ~/Plant_Counting_Data/Combined_Data --batch_size=64 --num_epochs 1 --model_name $MODEL_NAME

python validation_runner.py $MODEL_SAVE_PATH ~/Plant_Counting_Data/Combined_Data $SAVE_DIR $TEST_NAME CNN

cp ${SAVE_DIR}/${TEST_NAME} ~/Plant_Clump_Counting/all_test_results/

python compare_models.py ~/Plant_Clump_Counting/all_test_results --is_dir

