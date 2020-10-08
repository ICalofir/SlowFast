python tools/run_net.py \
	--cfg configs/custom/I3D_8x8_R50_train_expected_goals.yaml \
	DATA.PATH_TO_DATA_DIR /mnt/storage1/dissertation_dataset/expected_goals/configs \
	DATA.PATH_PREFIX /mnt/storage1/dissertation_dataset/expected_goals \
	MODEL.NUM_CLASSES 2 \
	CUSTOM_CONFIG.WEIGHT [0.65,2.18] \
	CUSTOM_CONFIG.TASK expected_goals \
	OUTPUT_DIR /mnt/storage1/logdir/slowfast/05_09_2020_expected_goals \
	TENSORBOARD.CLASS_NAMES_PATH /home/ionutc/Documents/Repositories/SlowFast/files/class_names_expected_goals.json \
	TENSORBOARD.CONFUSION_MATRIX.ENABLE True \
	TENSORBOARD.HISTOGRAM.ENABLE True \
	TENSORBOARD.HISTOGRAM.TOPK 2 \
	TRAIN.EVAL_PERIOD 1 \
	LOG_PERIOD 1 \
	SOLVER.MAX_EPOCH 50
