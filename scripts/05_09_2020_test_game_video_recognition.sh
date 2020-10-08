while IFS= read -r line
do
	GAMEPATH="$(echo $line | cut -d' ' -f1)"
	GAMECONFIGPATH="$(echo $line | cut -d' ' -f2)"
	echo $GAMEPATH
	echo $GAMECONFIGPATH

	python tools/run_net.py \
		--cfg configs/custom/I3D_8x8_R50_test.yaml \
		DATA.PATH_TO_DATA_DIR $GAMECONFIGPATH \
		DATA.PATH_PREFIX $GAMEPATH \
		TEST.CHECKPOINT_FILE_PATH /home/ionutc/Documents/Repositories/SlowFast/models/best_ckpt_video_recognition/checkpoints/checkpoint_epoch_00007.pyth \
		MODEL.NUM_CLASSES 3 \
		CUSTOM_CONFIG.WEIGHT [1.03,0.54,5.37] \
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

done < /mnt/storage1/dissertation_dataset/games/configs/test_vr.txt
