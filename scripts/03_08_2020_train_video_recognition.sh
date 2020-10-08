python tools/run_net.py \
	--cfg configs/custom/I3D_8x8_R50_train.yaml \
	DATA.PATH_TO_DATA_DIR /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/video_recognition/configs \
	DATA.PATH_PREFIX /home/ionutc/Documents/Repositories/Dissertation-2020/datasets/video_recognition \
	MODEL.NUM_CLASSES 3 \
	CUSTOM_CONFIG.WEIGHT [1.1,0.53,5.36] \
	OUTPUT_DIR models/03_08_2020_model_video_recognition
