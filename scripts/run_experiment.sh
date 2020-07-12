SEED1=$1
SEED2=$2
SEED3=$3

pythonpath=$4
logdir=$5
config=$6
id=$7

$pythonpath run.py --config $config --log_dir $logdir --seed $SEED1 --id $id &
# sleep 10s

$pythonpath run.py --config $config --log_dir $logdir --seed $SEED2 --id $id &
# sleep 10s

$pythonpath run.py --config $config --log_dir $logdir --seed $SEED3 --id $id &
# sleep 10s

wait