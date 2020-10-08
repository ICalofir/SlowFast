python tools/run_net.py \
	--cfg configs/custom/I3D_8x8_R50_test.yaml \
	DATA.PATH_TO_DATA_DIR /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/games/dump_20200815-141956330866/gt_videos_observations \
	DATA.PATH_PREFIX /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/games/dump_20200815-141956330866 \
	TEST.CHECKPOINT_FILE_PATH /home/ionutc/Documents/Repositories/SlowFast/models/best_ckpt_expected_goals/checkpoints/checkpoint_epoch_00049.pyth \
	MODEL.NUM_CLASSES 2 \
	CUSTOM_CONFIG.WEIGHT [0,0] \
	CUSTOM_CONFIG.TASK expected_goals \
	CUSTOM_CONFIG.TEST_TASK game \
	OUTPUT_DIR models/best_ckpt_expected_goals \
	TENSORBOARD.CLASS_NAMES_PATH /home/ionutc/Documents/Repositories/SlowFast/files/class_names_expected_goals.json \
	TENSORBOARD.CONFUSION_MATRIX.ENABLE False \
	TENSORBOARD.HISTOGRAM.ENABLE False \
	TENSORBOARD.HISTOGRAM.TOPK 2 \
	TENSORBOARD.MODEL_VIS.ENABLE False \
	TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS False \
	TENSORBOARD.MODEL_VIS.ACTIVATIONS False \
	TENSORBOARD.MODEL_VIS.INPUT_VIDEO False
