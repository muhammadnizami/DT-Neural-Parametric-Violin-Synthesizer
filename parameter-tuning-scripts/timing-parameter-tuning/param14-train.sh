python_cmd=python

timing_config='{"num_prev_context":3,"num_next_context":3,"max_depth":1000}'

timing_model_dir=model/parameter-tuning/timing-parameter-tuning/param14
num_folds=5

echo experiment timing with parameters $timing_config
for i in `seq 1 $num_folds`
do
	echo fold $i
	fold_timing_model_dir=$timing_model_dir/fold$i
	data_paths_file=data/lists/5-fold/train-dataset-f$i.txt
	mkdir -p $fold_timing_model_dir
	cmd="$python_cmd train.py --train-data-paths-file $data_paths_file \
		--timing-model-dir $fold_timing_model_dir --timing \
		--change-timing-config $timing_config"
	echo \>\>\> $cmd
	$cmd
done