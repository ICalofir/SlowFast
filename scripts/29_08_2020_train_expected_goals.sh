python tools/run_net.py \
	--cfg configs/custom/I3D_8x8_R50_train_expected_goals.yaml \
	DATA.PATH_TO_DATA_DIR /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/expected_goals/configs \
	DATA.PATH_PREFIX /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/expected_goals \
	MODEL.NUM_CLASSES 2 \
	CUSTOM_CONFIG.WEIGHT [0.65,2.18] \
	CUSTOM_CONFIG.TASK expected_goals \
	OUTPUT_DIR /mnt/storage1/models/29_08_2020_model_expected_goals \
	TENSORBOARD.CLASS_NAMES_PATH /home/ionutc/Documents/Repositories/SlowFast/files/class_names_expected_goals.json \
	TENSORBOARD.CONFUSION_MATRIX.ENABLE True \
	TENSORBOARD.HISTOGRAM.ENABLE True \
	TENSORBOARD.HISTOGRAM.TOPK 2 \
	TRAIN.EVAL_PERIOD 2 \
	LOG_PERIOD 1 \
	SOLVER.MAX_EPOCH 100
