SEED1=$1
SEED1=$2
SEED1=$3

pythonpath=$4
logdir=$5
config=$6
id=$7

$pythonpath run.py --config $config --log_dir $logdir --seed $SEED1 --id $id &

$pythonpath run.py --config $config --log_dir $logdir --seed $SEED2 --id $id &

$pythonpath run.py --config $config --log_dir $logdir --seed $SEED3 --id $id &