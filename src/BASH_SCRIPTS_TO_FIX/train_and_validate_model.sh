MODEL_NAME=EncoderCNN_ALLSynth #SimpleCNN_3_out_3_conv
SAVE_DIR=~/Plant_Emergence_Counting/Trained_Models/Counting/CNN/$MODEL_NAME

MODEL_SAVE_NAME="$MODEL_NAME.h5"
MODEL_JSON_NAME="$MODEL_NAME.json"
MODEL_SAVE_PATH="$SAVE_DIR/$MODEL_SAVE_NAME"
MODEL_JSON_PATH="$SAVE_DIR/$MODEL_JSON_NAME"
TEST_NAME="${MODEL_NAME}_test_scores.csv"

NUM_EPOCHS=1000

#python counting_CNN/train_runner.py ~/Plant_Counting_Data/synthetic2/ --model_save_dir $SAVE_DIR --validation_data_dir ~/Plant_Counting_Data/Combined_Data --batch_size=64 --num_epochs $NUM_EPOCHS --model_name $MODEL_NAME

python ../counting_models/encoder/train_runner.py ~/Plant_Counting_Data/synthetic2_train/ --model_save_dir $SAVE_DIR --validation_data_dir ~/Plant_Counting_Data/synthetic2_val --batch_size=32 --num_epochs $NUM_EPOCHS --model_name $MODEL_NAME

# python validation_runner.py $MODEL_JSON_PATH ~/Plant_Counting_Data/Combined_Data $SAVE_DIR $TEST_NAME CNN --path_to_weights $MODEL_SAVE_PATH
python ../validation_runner.py $MODEL_JSON_PATH ~/Plant_Counting_Data/synthetic2_val $SAVE_DIR $TEST_NAME ENCODER --path_to_weights $MODEL_SAVE_PATH

#cp ${SAVE_DIR}/${TEST_NAME} ~/Plant_Clump_Counting/all_test_results/

#python compare_models.py ~/Plant_Clump_Counting/all_test_results --is_dir

