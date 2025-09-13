echo "phobert_gg_v2 test on gg: "
CUDA_VISIBLE_DEVICES=1 python test.py --path data/merged/google_v2_json/ --type phobert --output_path ckpt/phobert_gg_v2/ --is_test

echo "vit5_gt test on gg: "
CUDA_VISIBLE_DEVICES=1 python test.py --path data/merged/google_v2_json/ --type t5 --output_path ckpt/vit5_gg_v2 --is_test

echo "mbert_gt test on gg: "
CUDA_VISIBLE_DEVICES=1 python test.py --path data/merged/google_v2_json/ --type bert --output_path ckpt/mbert_gg_v2/ --is_test

echo "bartpho_gt test on gg: "
CUDA_VISIBLE_DEVICES=1 python test.py --path data/merged/google_v2_json/ --type bartpho_w --output_path ckpt/bartpho_gg_v2/ --is_test

echo "xmlr_gt test on gg: "
CUDA_VISIBLE_DEVICES=1 python test.py --path data/merged/google_v2_json/ --type xlmroberta --output_path ckpt/xlmr_gg_v2/checkpoint-15670-epoch-10 --is_test