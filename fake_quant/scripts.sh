CUDA_VISIBLE_DEVICES=0 python main.py \
--model meta-llama/Llama-2-7b-hf \
--rotate --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --w_clip --w_rtn


CUDA_VISIBLE_DEVICES=0 python main.py \
--model meta-llama/Llama-2-7b-hf \
--rotate --eval_ppl


CUDA_VISIBLE_DEVICES=0 python main.py \
--model meta-llama/Llama-2-7b-hf \
--lr_fuse --eval_ppl

