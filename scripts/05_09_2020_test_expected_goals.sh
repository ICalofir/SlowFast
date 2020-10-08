python tools/run_net.py \
	--cfg configs/custom/I3D_8x8_R50_test.yaml \
	DATA.PATH_TO_DATA_DIR /mnt/storage1/dissertation_dataset/expected_goals/configs \
	DATA.PATH_PREFIX /mnt/storage1/dissertation_dataset/expected_goals \
	TEST.CHECKPOINT_FILE_PATH /home/ionutc/Documents/Repositories/SlowFast/models/best_ckpt_expected_goals/checkpoints/checkpoint_epoch_00027.pyth \
	MODEL.NUM_CLASSES 2 \
	CUSTOM_CONFIG.WEIGHT [0.65,2.2] \
	CUSTOM_CONFIG.TASK expected_goals \
	OUTPUT_DIR models/best_ckpt_expected_goals \
	TENSORBOARD.CLASS_NAMES_PATH /home/ionutc/Documents/Repositories/SlowFast/files/class_names_expected_goals.json \
	TENSORBOARD.CONFUSION_MATRIX.ENABLE True \
	TENSORBOARD.HISTOGRAM.ENABLE True \
	TENSORBOARD.HISTOGRAM.TOPK 2 \
	TENSORBOARD.MODEL_VIS.ENABLE True \
	TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS False \
	TENSORBOARD.MODEL_VIS.ACTIVATIONS False \
	TENSORBOARD.MODEL_VIS.INPUT_VIDEO False
