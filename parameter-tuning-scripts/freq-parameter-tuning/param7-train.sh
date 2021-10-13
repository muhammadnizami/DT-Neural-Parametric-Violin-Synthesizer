python_cmd=python

freq_config='{"dilation_factors":"1,2,4,1,2","corruption":"0.4","patience":"10000"}'

freq_model_dir=model/parameter-tuning/freq-parameter-tuning/param7
num_folds=5

echo experiment freq with parameters $freq_config
for i in `seq 1 $num_folds`
do
	echo fold $i
	fold_freq_model_dir=$freq_model_dir/fold$i
	data_paths_file=data/lists/5-fold/train-dataset-f$i.txt
	mkdir -p $fold_freq_model_dir
	cmd="$python_cmd train.py --train-data-paths-file $data_paths_file \
		--freq-model-dir $fold_freq_model_dir --freq
		--change-freq-config $freq_config"
	echo \>\>\> $cmd
	$cmd
done