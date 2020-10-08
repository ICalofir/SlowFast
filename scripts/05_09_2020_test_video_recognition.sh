python tools/run_net.py \
	--cfg configs/custom/I3D_8x8_R50_test.yaml \
	DATA.PATH_TO_DATA_DIR /mnt/storage1/dissertation_dataset/video_recognition/configs \
	DATA.PATH_PREFIX /mnt/storage1/dissertation_dataset/video_recognition \
	TEST.CHECKPOINT_FILE_PATH /home/ionutc/Documents/Repositories/SlowFast/models/best_ckpt_video_recognition/checkpoints/checkpoint_epoch_00007.pyth \
	MODEL.NUM_CLASSES 3 \
	CUSTOM_CONFIG.WEIGHT [1.03,0.54,5.37] \
	CUSTOM_CONFIG.TASK video_recognition \
	OUTPUT_DIR models/best_ckpt_video_recognition \
	TENSORBOARD.CLASS_NAMES_PATH /home/ionutc/Documents/Repositories/SlowFast/files/class_names_video_recognition.json \
	TENSORBOARD.CONFUSION_MATRIX.ENABLE True \
	TENSORBOARD.HISTOGRAM.ENABLE True \
	TENSORBOARD.HISTOGRAM.TOPK 3 \
	TENSORBOARD.MODEL_VIS.ENABLE True \
	TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS False \
	TENSORBOARD.MODEL_VIS.ACTIVATIONS False \
	TENSORBOARD.MODEL_VIS.INPUT_VIDEO False
