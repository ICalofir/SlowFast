python tools/run_net.py \
	--cfg configs/custom/I3D_8x8_R50_train.yaml \
	DATA.PATH_TO_DATA_DIR /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/video_recognition/configs \
	DATA.PATH_PREFIX /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/video_recognition \
	MODEL.NUM_CLASSES 3 \
	CUSTOM_CONFIG.WEIGHT [1.07,0.53,5.26] \
	OUTPUT_DIR models/04_08_2020_model_video_recognition \
	TENSORBOARD.CLASS_NAMES_PATH /home/ionutc/Documents/Repositories/SlowFast/files/class_names_video_recognition.json \
	TENSORBOARD.CONFUSION_MATRIX.ENABLE True \
	TENSORBOARD.HISTOGRAM.ENABLE True \
	TENSORBOARD.HISTOGRAM.TOPK 3 \
	TRAIN.EVAL_PERIOD 2 \
	LOG_PERIOD 1
