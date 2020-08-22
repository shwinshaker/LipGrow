##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash
debug=0 # 0 # 

model='resnet' # "preresnet" 
dataset='cifar10' # cifar10
depth=20 # 74 # 3*2 * num_blocks_per_layer + 2
grow=true
hooker='Lip'

# ------ grow setting -----------
mode='fixed' # 'adapt' # fixed
maxdepth=74 # if mode == adapt

# ----- fixed setting ---------------
dupEpoch=()
if [ "$grow" = true ] && [ "$mode" = 'fixed' ]; then
    dupEpoch=(60 110) # grow at xx epochs
fi

# ------ adapt setting --------------
thresh='1.4' # threshold to trigger grow
reserve=30 # reserved epochs for final model
window=10 # smooth window


# ----- regular hypers -----------
epochs=164
lr='0.5' # initial learning rate
scheduler='adacosine' # learning rate scheduler: cosine / constant / step
if [ "$grow" = true ]; then
    # if grow, no need to set learning rate scheduler
    schedule=() 
else
    # otherwise, set learning rate scheduler (if using step scheduler)
    schedule=(60 110) 
fi
gamma='0.1' # lr decaying factor, if using step lr scheduler
weight_decay='1e-4'
train_batch='128'
test_batch='100'

gpu_id='5' # For multiple gpu training, set like '1,2'
workers=4 # 4 * num gpus; or estimate by throughput
log_file="train.out"
suffix=""
prefix="Batch-Lip"

if (( debug > 0 )); then
    # debug mode - train a few epochs
    epochs=10
    schedule=() # 2 7)
    dupEpoch=(3 5)
    thresh='1.0'
    reserve=2
    window=3
fi

# set dir name
if [ "$grow" = true ]; then
    if [ "$mode" = 'fixed' ]; then
	if [ "$scheduler" = 'constant' ]; then
	    dir="$model-$depth-"$mode-"$(IFS='-'; printf '%s' "${dupEpoch[*]}")"-$operation-$scheduler-"lr=${lr//'.'/'-'}"
	else
	    dir="$model-$depth-"$mode-"$(IFS='-'; printf '%s' "${dupEpoch[*]}")"-$operation-$scheduler-"$(IFS='-'; printf '%s' "${schedule[*]}")"-"lr=${lr//'.'/'-'}"
	fi
    else
        dir="$model-$depth-$mode-$maxdepth-$operation-$scheduler"-"lr=${lr//'.'/'-'}-window=$window-reserve=$reserve-thresh=${thresh//'.'/'-'}"
    fi
else
    if [ "$scheduler" = 'constant' ]; then
	dir="$model-$depth-$scheduler-lr=${lr//'.'/'-'}"
    else
	dir="$model-$depth-$scheduler-"$(IFS='-'; printf '%s' "${schedule[*]}")"-lr=${lr//'.'/'-'}"
    fi
fi

if [ "$scheduler" = step ];then
    dir="$dir-gamma=${gamma//'.'/'-'}"
fi

if [ ! -z "$suffix" ];then
    dir=$dir'_'$suffix
fi

if [ ! -z "$prefix" ];then
    dir=$prefix-$dir
fi

if (( debug > 0 )); then
    dir="Debug-"$dir
fi

checkpoint="checkpoints/$dataset/$dir"
[[ -f $checkpoint ]] && rm $checkpoint
i=1
while [ -d $checkpoint ]; do
    echo '-----------------------------------------------------------------------------------------'
    ls $checkpoint
    tail -n 5 $checkpoint/train.out
    read -p "Checkpoint path $checkpoint already exists. Delete[d], Rename[r], or Terminate[*]? " ans
    case $ans in
	d ) rm -rf $checkpoint; break;;
	r ) checkpoint=${checkpoint%%_*}"_"$i;;
	* ) exit;;
    esac
    (( i++ ))
done
if [ ! -f $checkpoint ];then
    mkdir $checkpoint
fi
echo "Checkpoint path: "$checkpoint
echo 'Save main script to dir..'
cp launch.sh $checkpoint
cp train.py $checkpoint
cp -r utils $checkpoint
cp -r models $checkpoint

if [ "$grow" = true ]; then
    if (( debug > 0 )); then
	python train.py -d $dataset -a $model --grow --depth $depth --mode $mode --max-depth $maxdepth --epochs $epochs --grow-epoch "${dupEpoch[@]}" --threshold $thresh --window $window --reserve $reserve --hooker $hooker --scheduler $scheduler --schedule "${schedule[@]}" --gamma $gamma --wd $weight_decay --lr $lr --train-batch $train_batch --test-batch $test_batch --checkpoint "$checkpoint" --gpu-id "$gpu_id" --workers $workers --debug-batch-size $debug 2>&1 | tee "$checkpoint""/"$log_file
    else
	python train.py -d $dataset -a $model --grow --depth $depth --mode $mode --max-depth $maxdepth --epochs $epochs --grow-epoch "${dupEpoch[@]}" --threshold $thresh --window $window --reserve $reserve --hooker $hooker --scheduler $scheduler --schedule "${schedule[@]}" --gamma $gamma --wd $weight_decay --lr $lr --train-batch $train_batch --test-batch $test_batch --checkpoint "$checkpoint" --gpu-id "$gpu_id" --workers $workers --debug-batch-size $debug > "$checkpoint""/"$log_file 2>&1 &
    fi
else 
    if (( debug > 0 )); then
	python train.py -d $dataset -a $model --depth $depth --epochs $epochs --hooker $hooker --scheduler $scheduler --schedule "${schedule[@]}" --gamma $gamma --wd $weight_decay --lr $lr --train-batch $train_batch --test-batch $test_batch --checkpoint "$checkpoint" --gpu-id "$gpu_id" --workers $workers --debug-batch-size $debug | tee "$checkpoint""/"$log_file
    else
	python train.py -d $dataset -a $model --depth $depth --epochs $epochs --hooker $hooker --scheduler $scheduler --schedule "${schedule[@]}" --gamma $gamma --wd $weight_decay --lr $lr --train-batch $train_batch --test-batch $test_batch --checkpoint "$checkpoint" --gpu-id "$gpu_id" --workers $workers --debug-batch-size $debug > "$checkpoint""/"$log_file 2>&1 &
    fi
fi
pid=$!
echo "[$pid] [$gpu_id] [Path]: $checkpoint"
if (( debug == 0 )); then
    echo "s [$pid] [$gpu_id] $(date) [Path]: $checkpoint" >> log-cifar.txt
fi

