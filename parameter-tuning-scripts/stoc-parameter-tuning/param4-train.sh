python_cmd=python

stoc_config='{"dilation_factors":"1,2,4,1,2","corruption":"0.01"}'

stoc_model_dir=model/parameter-tuning/stoc-parameter-tuning/param4
num_folds=5

echo experiment stoc with parameters $stoc_config
for i in `seq 1 $num_folds`
do
	echo fold $i
	fold_stoc_model_dir=$stoc_model_dir/fold$i
	data_paths_file=data/lists/5-fold/train-dataset-f$i.txt
	mkdir -p $fold_stoc_model_dir
	cmd="$python_cmd train.py --train-data-paths-file $data_paths_file \
		--stoc-model-dir $fold_stoc_model_dir --stoc
		--change-stoc-config $stoc_config"
	echo \>\>\> $cmd
	$cmd
done