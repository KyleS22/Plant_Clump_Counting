

TRAIN_DIR=~/Plant_Counting_Data/train/
TEST_DIR=~/Plant_Counting_Data/test/

SYNTH_TRAIN_DIR=~/Plant_Counting_Data/synthetic_train/
SYNTH_TEST_DIR=~/Plant_Counting_Data/synthetic_test/

NO_SYNTH_NAME=NoSynthVGG_L2
NO_SYNTH_DIR=./$NO_SYNTH_NAME

SYNTH_NAME=Synthetic_VGG_L2
SYNTH_DIR=./$SYNTH_NAME
TRAINED_SYNTH_DIR=./Trained_$SYNTH_NAME


python model.py $TRAIN_DIR $TEST_DIR --batch_size 32 --num_epochs 500 --model_name $NO_SYNTH_NAME --model_save_dir $NO_SYNTH_DIR
python model.py $SYNTH_TRAIN_DIR  $SYNTH_TEST_DIR --batch_size 32 --num_epochs 500 --model_name $SYNTH_NAME --model_save_dir $SYNTH_DIR
python model.py ~/Plant_Counting_Data/train/ ~/Plant_Counting_Data/test/ --model_load_dir $SYNTH_DIR --batch_size 32 --num_epochs 500 --model_name $SYNTH_NAME --model_save_dir $TRAINED_SYNTH_DIR
python summarize_models.py
