#!/usr/bin/env bash

show_help() {
cat << EOF
Usage: ${0##*/} [-h] [--input INPUT] [--save DIRECTORY]

This script processes sequencing data with optional aheads including basecalling, barcoding, and compression.

    -h, --help             display this help and exit
    --input  INPUT         specify the table containing the relative abundance and technical values
    --save   DIRECTORY     specify save directory
    --epochs EPOCHS        specify the number of epochs for training
    --layers LAYERS        specify the number of additional layers
    --max-layers MAX_LAYERS  specify the maximum number of layers for training
    --batch  BATCH         specify the batch size for training
    --unroll UNROLL        specify the unroll option for LSTM model, default: 0
    --back   BACKGROUND    the number of background days
    --nodes  NODES         specify the number of units per hidden layer
    --ahead  AHEAD          specify the number of background days per ahead
    --target TARGET        specify loss value for monitoring callback
    --rounds ROUNDS        specify the number of rounds for training
    --logs   LOGS          specify the name of log file.
    --add    ADD           add a particular number for the propper round of training
    --future FUTURE        specify a number of aheads ahead for predicting
    --stop   STOP          specify the early stop for the loss
    --lrate  LEARNING RATE specify learning rate for the optimizer, default: 0.001
    --model  MODEL NAME
    --device CUDA DEVICE

Examples:
    ${0##*/} --input /netscratch/dep_psl/grp_rgo/vm/Tables_583d_seed1234_l1k --save /netscratch/dep_psl/grp_rgo/vm/LSTM_v3 --model model --device cuda:2 --batch 2048 --layers 3 --max-layers 3 --nodes 256 --epochs 2000 --rounds 10 --lrate 0.001
EOF
}

# Initialize variables with default values
epochs=100
layers=1
max_layers=4
batch=1024
back=128
nodes=128
ahead=10
rounds=1
input=""
add=0
future=0
lrate=0.001
model="model"
device="cuda:0"
warm=3

while (( "$#" )); do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --input)
      input="$2"
      shift 2
      ;;
    --save)
      save="$2"
      shift 2
      ;;
    --epochs)
      epochs="$2"
      shift 2
      ;;
    --layers)
      layers="$2"
      shift 2
      ;;
    --max-layers)
      max_layers="$2"
      shift 2
    ;;  
    --batch)
      batch="$2"
      shift 2
      ;;
    --back)
      back="$2"
      shift 2
    ;;
    --nodes)
      nodes="$2"
      shift 2
    ;;
    --unroll)
      unroll="$2"
      shift 2
    ;;
    --ahead)
      ahead="$2"
      shift 2
    ;;
    --rounds)
      rounds="$2"
      shift 2
    ;;
    --model)
      model="$2"
      shift 2
    ;;
    --logs)
      logs="$2"
      shift 2
    ;;
    --add)
      add="$2"
      shift 2
    ;;
    --future)
      future="$2"
      shift 2
    ;;
    --lrate)
      lrate="$2"
      shift 2
    ;;
    --device)
      device="$2"
      shift 2
    ;;
    --warm)
      warm="$2"
      shift 2
    ;;
    --)
      shift
      break
      ;;
    -*|--*=)
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *)
      echo "Error: Unsupported argument $1" >&2
      exit 1
      ;;
  esac
done

for (( i=1; i<=rounds; i++ ))
do
    current_layers=$layers
    for (( ; current_layers<=max_layers; current_layers++ ))
    do
        new_i=$((i + add))
        echo "Round of training ${new_i}"

        log_file="${logs}.${new_i}.txt"
        model_name="${model}_${nodes}_${current_layers}"

        if [ -z "$input" ]
        then
            python run_full.py -s "$save" -v "$model_name" --device "$device" -q "$new_i" -b "$batch" -l "$current_layers" -n "$nodes" -d "$back" -r "$lrate" -e "$epochs"
        else
            python run_full.py -t "$input" -s "$save" -v "$model_name" --device "$device" -l "$current_layers" -e "$epochs" -b "$batch" -n "$nodes" -p "$ahead" -d "$back" -q "$new_i" -r "$lrate"
        fi

        echo "Finish round #${new_i}"
    done
done