python_cmd=python

mag_config='{"dilation_factors":"1,2,4","corruption":"0.4"}'

mag_model_dir=model/parameter-tuning/mag-parameter-tuning/param3
num_folds=5

echo experiment mag with parameters $mag_config
for i in `seq 1 $num_folds`
do
	echo fold $i
	fold_mag_model_dir=$mag_model_dir/fold$i
	data_paths_file=data/lists/5-fold/train-dataset-f$i.txt
	mkdir -p $fold_mag_model_dir
	cmd="$python_cmd train.py --train-data-paths-file $data_paths_file \
		--mag-model-dir $fold_mag_model_dir --mag
		--change-mag-config $mag_config"
	echo \>\>\> $cmd
	$cmd
done