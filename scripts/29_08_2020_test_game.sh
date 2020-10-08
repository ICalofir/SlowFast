python tools/run_net.py \
	--cfg configs/custom/I3D_8x8_R50_test.yaml \
	DATA.PATH_TO_DATA_DIR /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/games/dump_20200815-141956330866/sliding_window_videos/sliding_window_videos_information_length_15 \
	DATA.PATH_PREFIX /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/games/dump_20200815-141956330866 \
	TEST.CHECKPOINT_FILE_PATH /home/ionutc/Documents/Repositories/SlowFast/models/best_ckpt_video_recognition/checkpoints/checkpoint_epoch_00002.pyth \
	MODEL.NUM_CLASSES 3 \
	CUSTOM_CONFIG.WEIGHT [1.05,0.54,5.33] \
	CUSTOM_CONFIG.TASK video_recognition \
	CUSTOM_CONFIG.TEST_TASK game \
	OUTPUT_DIR models/best_ckpt_video_recognition \
	TENSORBOARD.CLASS_NAMES_PATH /home/ionutc/Documents/Repositories/SlowFast/files/class_names_video_recognition.json \
	TENSORBOARD.CONFUSION_MATRIX.ENABLE False \
	TENSORBOARD.HISTOGRAM.ENABLE False \
	TENSORBOARD.HISTOGRAM.TOPK 3 \
	TENSORBOARD.MODEL_VIS.ENABLE False \
	TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS False \
	TENSORBOARD.MODEL_VIS.ACTIVATIONS False \
	TENSORBOARD.MODEL_VIS.INPUT_VIDEO False
