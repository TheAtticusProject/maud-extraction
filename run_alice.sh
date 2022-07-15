THREADS=1

batch_size=16

# BLURB:
# Because we reduced the batch size by 1/3, we do the same with lr.

for run_num in 1 2; do
  for vat_loss_weight in 1.0 0.5; do
    # python $pdb train.py \
    # srun -p jsteinhardt --gres=gpu:A5000:1 \
    $echo srun -p jsteinhardt --gres=gpu:A100:1 -c $THREADS -w balrog --comment=cuad_$vat_loss_weight \
    python $pdb train.py --alice \
      --thread $THREADS \
            --output_dir ./train_models/roberta-base-alice/loss_$vat_loss_weight/run_$run_num \
            --model_type roberta \
            --model_name_or_path roberta-base \
            --train_file ./data/train_separate_questions.json \
            --predict_file ./data/test.json \
            --do_train \
            --do_eval \
            --version_2_with_negative \
            --learning_rate 3e-5 \
            --num_train_epochs 4 \
            --vat-loss-weight $vat_loss_weight \
            --per_gpu_eval_batch_size=$batch_size  \
            --per_gpu_train_batch_size=$batch_size \
            --max_seq_length 512 \
            --max_answer_length 512 \
            --doc_stride 256 \
            --save_steps 1000 \
            --n_best_size 20 \
            --overwrite_output_dir  &
  done
done
wait
