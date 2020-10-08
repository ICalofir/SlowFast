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
		TEST.CHECKPOINT_FILE_PATH /home/ionutc/Documents/Repositories/SlowFast/models/best_ckpt_expected_goals/checkpoints/checkpoint_epoch_00027.pyth \
		MODEL.NUM_CLASSES 2 \
		CUSTOM_CONFIG.WEIGHT [0.65,2.2] \
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

done < /mnt/storage1/dissertation_dataset/games/configs/test_eg.txt
