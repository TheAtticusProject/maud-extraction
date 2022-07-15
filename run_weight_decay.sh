# THREADS=32
# THREADS=2
THREADS=1
n_runs=3
# echo="echo"

for weight_decay in 1e-3 1e-2 1e-1; do
	for run in $(seq $n_runs); do
		$echo srun -p jsteinhardt --gres=gpu:A100:1 -c $THREADS -w balrog \
		python train.py \
			--weight_decay $weight_decay \
			--thread $THREADS \
			--output_dir ./train_models/roberta-base-wd/weight_decay=$weight_decay/run=$run \
			--model_type roberta \
			--model_name_or_path roberta-base \
			--train_file ./data/train_separate_questions.json \
			--predict_file ./data/test.json \
			--do_train \
			--do_eval \
			--version_2_with_negative \
			--learning_rate 1e-4 \
			--num_train_epochs 4 \
			--per_gpu_eval_batch_size=40  \
			--per_gpu_train_batch_size=40 \
			--max_seq_length 512 \
			--max_answer_length 512 \
			--doc_stride 256 \
			--save_steps 1000 \
			--n_best_size 20 \
			--overwrite_output_dir  &
	done
done
wait
