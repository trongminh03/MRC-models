# python train.py --path merged/google_json/ --type bert --model VietAI/vit5-base --output_path ckpt/vit5_asembly/ --device 1 &
# python train.py --path merged/asembly_json/ --type phobert --model vinai/phobert-base --output_path ckpt/phobert_asembly/ --device 1 --max_seq_length 256 &
# python train.py --path merged/asembly_json/ --type bartpho_w --model vinai/bartpho-word-base --output_path ckpt/bartpho_asembly/ --device 0 &
# python train.py --path merged/novo_json/ --type bert --model google-bert/bert-base-multilingual-cased --output_path ckpt/mbert_novo/ --device 0

#python train.py --path merged/google_v2_json/ --type xlmroberta --model FacebookAI/xlm-roberta-base --output_path ckpt/xlmr_gg --device 0
#python train.py --path merged/asembly_json/ --type xlmroberta --model FacebookAI/xlm-roberta-base --output_path ckpt/xlmr_assembly --device 0 &
#python train.py --path merged/novo_json/ --type xlmroberta --model FacebookAI/xlm-roberta-base --output_path ckpt/xlmr_novo --device 1 &
#python train.py --path /data/ViSQA/data/json/ --type xlmroberta --model FacebookAI/xlm-roberta-base --output_path ckpt/xlmr --device 1


python train.py --path merged/google_v2_json/ --type phobert --model vinai/phobert-base --output_path ckpt/phobert_gg_v2/ --device 1 --max_seq_length 256 &
python train.py --path merged/google_v2_json/ --type xlmroberta --model FacebookAI/xlm-roberta-base --output_path ckpt/xlmr_gg_v2/ --device 0 &
python train.py --path merged/google_v2_json/ --type bert --model google-bert/bert-base-multilingual-cased --output_path ckpt/mbert_gg_v2/ --device 0 &
python train.py --path merged/google_v2_json/ --type bartpho_w --model vinai/bartpho-word-base --output_path ckpt/bartpho_gg_v2/ --device 0 &
python train.py --path merged/google_v2_json/ --type t5 --model VietAI/vit5-base --output_path ckpt/vit5_gg_v1/ --device 1
#
#
#python test.py --path merged/asembly_json/ --type xlmroberta --output_path ckpt/xlmr_assembly/checkpoint-15670-epoch-10 --is_test
#python test.py --path merged/novo_json/ --type xlmroberta --output_path ckpt/xlmr_novo/checkpoint-14380-epoch-10 --is_test
#python test.py --path merged/google_json/ --type xlmroberta --output_path ckpt/xlmr/checkpoint-24310-epoch-10 --is_test
