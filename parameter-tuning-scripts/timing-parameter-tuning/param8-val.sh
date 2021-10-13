python_cmd=python

timing_config='{"num_prev_context":1,"num_next_context":1,"max_depth":50}'

timing_model_dir=model/parameter-tuning/timing-parameter-tuning/param8
output_dir=outputs/parameter-tuning/timing-parameter-tuning/param8/
num_folds=5

echo experiment timing with parameters $timing_config
echo VALIDATION
# for i in `seq 1 $num_folds`
# do
# 	echo fold $i
# 	fold_timing_model_dir=$timing_model_dir/fold$i/
# 	data_paths_file=data/lists/5-fold/val-dataset-f$i.txt
# 	mkdir -p $fold_timing_model_dir
# 	mkdir -p $output_dir
# 	cmd="$python_cmd generate_batch.py --data-paths-file $data_paths_file \
# 		--output-dir $output_dir \
# 		--timing-model-dir $fold_timing_model_dir --only timing\
# 		--change-timing-config $timing_config"
# 	echo \>\>\> $cmd
# 	$cmd
# done

cmd="$python_cmd eval.py --data-paths-file data/lists/train-dataset.txt \
	--output-dir-timing $output_dir"
echo \>\>\> $cmd
$cmd