python tools/run_net.py \
	--cfg configs/custom/I3D_8x8_R50_test.yaml \
	DATA.PATH_TO_DATA_DIR /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/video_recognition/configs \
	DATA.PATH_PREFIX /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/video_recognition \
	MODEL.NUM_CLASSES 3 \
	TEST.CHECKPOINT_FILE_PATH /home/ionutc/Documents/Repositories/Dissertation-2020/models/01_06_2020_model_video_recognition/checkpoints/checkpoint_epoch_00010.pyth
