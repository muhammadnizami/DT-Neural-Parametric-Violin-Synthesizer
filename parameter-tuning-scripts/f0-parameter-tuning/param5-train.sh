python_cmd=python

f0_config='{"dilation_factors":"1,2,4,8,16,32,1,2,4,8,16","corruption":"0.4","patience":"10000"}'

f0_model_dir=model/parameter-tuning/f0-parameter-tuning/param5
num_folds=5

echo experiment f0 with parameters $f0_config
for i in `seq 1 $num_folds`
do
	echo fold $i
	fold_f0_model_dir=$f0_model_dir/fold$i
	data_paths_file=data/lists/5-fold/train-dataset-f$i.txt
	mkdir -p $fold_f0_model_dir
	cmd="$python_cmd train.py --train-data-paths-file $data_paths_file \
		--f0-model-dir $fold_f0_model_dir --f0
		--change-f0-config $f0_config"
	echo \>\>\> $cmd
	$cmd
done