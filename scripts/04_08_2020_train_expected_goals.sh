python tools/run_net.py \
	--cfg configs/custom/I3D_8x8_R50_train.yaml \
	DATA.PATH_TO_DATA_DIR /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/expected_goals/configs \
	DATA.PATH_PREFIX /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/expected_goals \
	MODEL.NUM_CLASSES 2 \
	CUSTOM_CONFIG.WEIGHT [0.65,2.19] \
	OUTPUT_DIR models/04_08_2020_model_expected_goals \
	TENSORBOARD.CLASS_NAMES_PATH /home/ionutc/Documents/Repositories/SlowFast/files/class_names_expected_goals.json \
	TENSORBOARD.CONFUSION_MATRIX.ENABLE True \
	TENSORBOARD.HISTOGRAM.ENABLE True \
	TENSORBOARD.HISTOGRAM.TOPK 2 \
	TRAIN.EVAL_PERIOD 2 \
	LOG_PERIOD 1
