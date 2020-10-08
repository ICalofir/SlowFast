python tools/run_net.py \
	--cfg configs/custom/I3D_8x8_R50_train_video_recognition.yaml \
	DATA.PATH_TO_DATA_DIR /mnt/storage1/dissertation_dataset/video_recognition/configs \
	DATA.PATH_PREFIX /mnt/storage1/dissertation_dataset/video_recognition \
	MODEL.NUM_CLASSES 3 \
	CUSTOM_CONFIG.WEIGHT [1.03,0.54,5.37] \
	CUSTOM_CONFIG.TASK video_recognition \
	OUTPUT_DIR /mnt/storage1/logdir/slowfast/05_09_2020_video_recognition \
	TENSORBOARD.CLASS_NAMES_PATH /home/ionutc/Documents/Repositories/SlowFast/files/class_names_video_recognition.json \
	TENSORBOARD.CONFUSION_MATRIX.ENABLE True \
	TENSORBOARD.HISTOGRAM.ENABLE True \
	TENSORBOARD.HISTOGRAM.TOPK 3 \
	TRAIN.EVAL_PERIOD 1 \
	LOG_PERIOD 1 \
	SOLVER.MAX_EPOCH 50
