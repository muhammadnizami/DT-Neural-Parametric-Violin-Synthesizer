python_cmd=python

mag_config='{"dilation_factors":"1,2,4","corruption":"0.4"}'

mag_model_dir=model/parameter-tuning/mag-parameter-tuning/param3
mag_output_dir=outputs/parameter-tuning/mag-parameter-tuning/param8
output_dir=$mag_output_dir
num_folds=5

freq_config='{"dilation_factors":"1,2,4","corruption":"0.4"}'

freq_model_dir=model/parameter-tuning/freq-parameter-tuning/param2
freq_model_output_dir=outputs/parameter-tuning/freq-parameter-tuning/param2
freq_data_dir=data/hps

f0_config='{"dilation_factors":"1,2,4,8,16,32,64,1,2,4,8,16,32","corruption":"0.01"}'
f0_model_dir=model/parameter-tuning/f0-parameter-tuning/param3
f0_model_output_dir=outputs/parameter-tuning/f0-parameter-tuning/param3

timing_config='{"num_prev_context":3,"num_next_context":3,"max_depth":600}'
timing_model_dir=model/parameter-tuning/timing-parameter-tuning/param2.5
timing_model_output_dir=outputs/parameter-tuning/timing-parameter-tuning/param2.5/
timing_data_dir=data/timings

echo experiment mag with parameters $timing_config
echo VALIDATION use prev model out
output_dir_p=$output_dir/use-prev-out/
freq_model_output_dir_p=$freq_model_output_dir/use-prev-out/
for i in `seq 1 $num_folds`
do
	echo fold $i
	fold_timing_model_dir=$timing_model_dir/fold$i/
	fold_freq_model_dir=$freq_model_dir/fold$i/
	fold_mag_model_dir=$mag_model_dir/fold$i/
	data_paths_file=data/lists/5-fold/val-dataset-f$i.txt
	mkdir -p $fold_timing_model_dir
	mkdir -p $output_dir_p
	cmd="$python_cmd generate_batch.py --data-paths-file $data_paths_file \
		--output-dir $output_dir_p \
		--timing-model-dir $fold_timing_model_dir --only mag\
        --mag-model-dir $fold_mag_model_dir\
        --change-mag-config $mag_config\
		--freq-model-dir $fold_freq_model_dir\
        --change-timing-config $timing_config\
        --change-freq-config $freq_config\
		--existing-data-dir $freq_model_output_dir_p\
		--existing-timing-dir $timing_model_output_dir"
	echo \>\>\> $cmd
	$cmd
done

echo VALIDATION use data out
output_dir_d=$output_dir/use-data-out/
freq_model_output_dir_d=$freq_model_output_dir/use-data-out/
for i in `seq 1 $num_folds`
do
	echo fold $i
	fold_timing_model_dir=$timing_model_dir/fold$i/
	fold_freq_model_dir=$freq_model_dir/fold$i/
	fold_mag_model_dir=$mag_model_dir/fold$i/
	data_paths_file=data/lists/5-fold/val-dataset-f$i.txt
	mkdir -p $fold_timing_model_dir
	mkdir -p $output_dir_d
	cmd="$python_cmd generate_batch.py --data-paths-file $data_paths_file \
		--output-dir $output_dir_d \
		--timing-model-dir $fold_timing_model_dir --only mag\
        --mag-model-dir $fold_mag_model_dir\
        --change-mag-config $mag_config\
		--freq-model-dir $fold_freq_model_dir\
        --change-timing-config $timing_config\
        --change-freq-config $freq_config\
		--existing-data-dir $freq_data_dir\
		--existing-timing-dir $timing_data_dir"
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
