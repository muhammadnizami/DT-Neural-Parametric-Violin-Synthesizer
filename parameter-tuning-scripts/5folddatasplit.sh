num_folds=$(echo 5)

IFS=$'\n' instances=($(cat data/lists/train-dataset.txt))
SUM_TIME=$(echo 0)
for i in `seq 0 $((${#instances[@]}-1))`;
do
	instance=${instances[$i]}
	timingfile=$(echo $instance | cut -f2)
	timing=$(cat $timingfile)
	time=$(echo $timing | rev | cut -d' ' -f 1 | rev)
	SUM_TIME=$(echo $SUM_TIME + ${time} | bc)
	times[$i]=$time
done;
VAL_TIME=$(echo $SUM_TIME/$num_folds | bc)

num_assigned=$(echo 0)
declare -a val_fold_assignment=( $(for i in `seq 0 $((${#instances[@]}-1))`; do echo 0; done) )

for i in `seq 1 $num_folds`
do
	fold_val_time[$i]=$(echo 0)
	until (( $(echo "${fold_val_time[$i]} >= $VAL_TIME || $num_assigned >= ${#instances[@]}" | bc))); do
		selected_idx=$(($RANDOM%${#instances[@]}))
		if (($(echo ${val_fold_assignment[$selected_idx]} == 0 | bc)))
		then
			fold_val_time[$i]=$(echo ${fold_val_time[$i]}+${times[$selected_idx]} | bc)
			val_fold_assignment[$selected_idx]=$i
			num_assigned=$(( $num_assigned + 1 ))
		fi
		if (($(echo ${fold_val_time[$i]} > $VAL_TIME | bc)))
		then
			if (($(echo ${fold_val_time[$i]} - $VAL_TIME > $VAL_TIME - (${fold_val_time[$i]} - ${times[$selected_idx]}) | bc)))
			then
				val_fold_assignment[$selected_idx]=0
				fold_val_time[$i]=$(echo ${fold_val_time[$i]} - ${times[$selected_idx]} | bc)
				num_assigned=$(( $num_assgined - 1 ))
			fi
		fi
	done
	echo fold $i with split $(echo ${SUM_TIME} - ${fold_val_time[$i]} | bc)	${fold_val_time[$i]}
done;

for i in `seq 1 $num_folds`
do
	trainfilename=data/lists/5-fold/train-dataset-f$i.txt
	valfilename=data/lists/5-fold/val-dataset-f$i.txt
	> $trainfilename
	> $valfilename
	for j in `seq 0 $((${#instances[@]}-1))`
	do
		if [ ${val_fold_assignment[$j]} -eq $i ]
		then
			echo ${instances[$j]} >> $valfilename
		else
			echo ${instances[$j]} >> $trainfilename
		fi
	done
done