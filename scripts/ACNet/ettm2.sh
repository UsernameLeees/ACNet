#!/bin/bash

model_name=ACNet

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

seq_len=96

# 定义网格搜索的超参数（基于ETTm2的默认参数调整）
train_epochs_list=(30 50)
patience_list=(5 7)
dropout_list=(0.7 0.8)
learning_rate_list=(0.0005 0.001 0.005)
random_seed_list=(2024)

# 预测长度
pred_len_list=(96 192 336 720)

# 创建日志目录
log_dir="./logs/grid_search_ETTm2_$(date +%Y%m%d_%H%M%S)"
mkdir -p $log_dir

# 创建总日志文件
summary_log="$log_dir/summary.log"
touch $summary_log

# 计数器，用于跟踪进度
total_experiments=0
current_experiment=0

# 计算总实验数
for pred_len in "${pred_len_list[@]}"; do
  for train_epochs in "${train_epochs_list[@]}"; do
    for patience in "${patience_list[@]}"; do
      for dropout in "${dropout_list[@]}"; do
        for learning_rate in "${learning_rate_list[@]}"; do
          for random_seed in "${random_seed_list[@]}"; do
            ((total_experiments++))
          done
        done
      done
    done
  done
done

echo "Total experiments to run: $total_experiments" | tee -a $summary_log
echo "Log directory: $log_dir" | tee -a $summary_log
echo "" | tee -a $summary_log

# 运行网格搜索
for pred_len in "${pred_len_list[@]}"; do
  for train_epochs in "${train_epochs_list[@]}"; do
    for patience in "${patience_list[@]}"; do
      for dropout in "${dropout_list[@]}"; do
        for learning_rate in "${learning_rate_list[@]}"; do
          for random_seed in "${random_seed_list[@]}"; do
            ((current_experiment++))
            
            # 创建实验日志文件名
            exp_log="$log_dir/exp_${current_experiment}_pred${pred_len}_ep${train_epochs}_pat${patience}_drop${dropout}_lr${learning_rate}_seed${random_seed}.log"
            
            echo "==================================" | tee -a $summary_log
            echo "Experiment $current_experiment/$total_experiments" | tee -a $summary_log
            echo "pred_len=$pred_len, epochs=$train_epochs, patience=$patience, dropout=$dropout, lr=$learning_rate, seed=$random_seed" | tee -a $summary_log
            echo "Log file: $exp_log" | tee -a $summary_log
            echo "==================================" | tee -a $summary_log
            
            # 运行实验并重定向输出到日志文件
            python -u run.py \
              --is_training 1 \
              --root_path $root_path_name \
              --data_path $data_path_name \
              --model_id $model_id_name'_'$seq_len'_'$pred_len'_ep'$train_epochs'_pat'$patience'_drop'$dropout'_lr'$learning_rate'_seed'$random_seed \
              --model $model_name \
              --data $data_name \
              --features M \
              --seq_len $seq_len \
              --pred_len $pred_len \
              --enc_in 7 \
              --cycle 96 \
              --train_epochs $train_epochs \
              --patience $patience \
              --dropout $dropout \
              --itr 1 \
              --batch_size 256 \
              --learning_rate $learning_rate \
              --random_seed $random_seed \
              2>&1 | tee $exp_log
            
            echo "" | tee -a $summary_log
          done
        done
      done
    done
  done
done

echo "All experiments completed!" | tee -a $summary_log
echo "Results saved in: $log_dir" | tee -a $summary_log