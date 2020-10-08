python tools/run_net.py \
	--cfg configs/Kinetics/I3D_8x8_R50.yaml \
	DATA.PATH_TO_DATA_DIR "/home/ionutc/Documents/Repositories/Dissertation-2020/datasets/video_recognition" \
	DATA.PATH_PREFIX "/home/ionutc/Documents/Repositories/Dissertation-2020/datasets/video_recognition" \
	NUM_GPUS 1 \
	TRAIN.BATCH_SIZE 4 \
	MODEL.NUM_CLASSES 400 \
	TRAIN.CHECKPOINT_FILE_PATH "/home/ionutc/Downloads/I3D_8x8_R50.pkl" \
	TRAIN.CHECKPOINT_TYPE caffe2 \
	TEST.CHECKPOINT_FILE_PATH "/home/ionutc/Documents/Repositories/SlowFast/freeze_model/checkpoint_epoch_00151.pyth" \
	TRAIN.ENABLE False \
	TEST.BATCH_SIZE 4
