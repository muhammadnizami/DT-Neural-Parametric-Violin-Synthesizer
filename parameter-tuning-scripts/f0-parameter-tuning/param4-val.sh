python_cmd=python

f0_config='{"dilation_factors":"1,2,4,8,16,32,1,2,4,8,16","corruption":"0.01"}'
f0_model_dir=model/parameter-tuning/f0-parameter-tuning/param4
output_dir=outputs/parameter-tuning/f0-parameter-tuning/param4
num_folds=5

timing_config='{"num_prev_context":3,"num_next_context":3,"max_depth":600}'
timing_model_dir=model/parameter-tuning/timing-parameter-tuning/param2.5
timing_model_output_dir=outputs/parameter-tuning/timing-parameter-tuning/param2.5/
timing_data_dir=data/timings

echo experiment f0 with parameters $timing_config
echo VALIDATION use prev model out
output_dir_p=$output_dir/use-prev-out/
for i in `seq 1 $num_folds`
do
	echo fold $i
	fold_timing_model_dir=$timing_model_dir/fold$i/
	fold_f0_model_dir=$f0_model_dir/fold$i/
	data_paths_file=data/lists/5-fold/val-dataset-f$i.txt
	mkdir -p $fold_timing_model_dir
	mkdir -p $output_dir_p
	cmd="$python_cmd generate_batch.py --data-paths-file $data_paths_file \
		--output-dir $output_dir_p \
		--timing-model-dir $fold_timing_model_dir --only f0\
		--f0-model-dir $fold_f0_model_dir\
        --change-timing-config $timing_config\
        --change-f0-config $f0_config\
		--existing-data-dir $timing_model_output_dir"
	echo \>\>\> $cmd
	$cmd
done

echo VALIDATION use data out
output_dir_d=$output_dir/use-data-out/
for i in `seq 1 $num_folds`
do
	echo fold $i
	fold_timing_model_dir=$timing_model_dir/fold$i/
	fold_f0_model_dir=$f0_model_dir/fold$i/
	data_paths_file=data/lists/5-fold/val-dataset-f$i.txt
	mkdir -p $fold_timing_model_dir
	mkdir -p $output_dir_d
	cmd="$python_cmd generate_batch.py --data-paths-file $data_paths_file \
		--output-dir $output_dir_d \
		--timing-model-dir $fold_timing_model_dir --only f0\
		--f0-model-dir $fold_f0_model_dir\
        --change-timing-config $timing_config\
        --change-f0-config $f0_config\
		--existing-data-dir $timing_data_dir"
	echo \>\>\> $cmd
	$cmd
done

echo 'use prev 1'
cmd="$python_cmd eval.py --data-paths-file data/lists/train-dataset.txt \
	--output-dir-hps $output_dir_p"
echo \>\>\> $cmd
$cmd

echo 'use data'
cmd="$python_cmd eval.py --data-paths-file data/lists/train-dataset.txt \
	--output-dir-hps $output_dir_d"
echo \>\>\> $cmd
$cmd
