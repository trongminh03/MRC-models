echo "phobert_gt test on assembly: "
CUDA_VISIBLE_DEVICES=1 python test.py --path /data/trongminh/VN-SQA/data/merged/google_v2_json/ --type phobert --output_path /data/trongminh/VN-SQA/ckpt/phobert_asembly/ --is_test

echo "vit5_gt test on assembly: "
CUDA_VISIBLE_DEVICES=1 python test.py --path /data/trongminh/VN-SQA/data/merged/google_v2_json/ --type t5 --output_path /data/trongminh/VN-SQA/ckpt/vit5_asembly/ --is_test

echo "mbert_gt test on assembly: "
CUDA_VISIBLE_DEVICES=1 python test.py --path /data/trongminh/VN-SQA/data/merged/google_v2_json/ --type bert --output_path /data/trongminh/VN-SQA/ckpt/mbert_asembly/ --is_test

echo "bartpho_gt test on assembly: "
CUDA_VISIBLE_DEVICES=1 python test.py --path /data/trongminh/VN-SQA/data/merged/google_v2_json/ --type bartpho_w --output_path /data/trongminh/VN-SQA/ckpt/bartpho_asembly/ --is_test

echo "xmlr_gt test on assembly: "
CUDA_VISIBLE_DEVICES=1 python test.py --path /data/trongminh/VN-SQA/data/merged/google_v2_json/ --type xlmroberta --output_path /data/trongminh/VN-SQA/ckpt/xlmr_assembly/checkpoint-15670-epoch-10 --is_test

#echo "mbert_gg test on gg: "
#python test.py --path /data/trongminh/VN-SQA/data/merged/google_json/ --type bert --output_path /data/trongminh/VN-SQA/ckpt/mbert_google/ --is_test

#echo "bartpho_gt test on asembly: "
#python test.py --path /data/trongminh/VN-SQA/data/merged/asembly_json/ --type bartpho_w --output_path /data/ViSQA/ckpt/bartpho/ --is_test



#echo "mbert_nova test on nova: "
#python test.py --path /data/trongminh/VN-SQA/data/merged/novo_json/ --type bert --output_path /data/trongminh/VN-SQA/ckpt/mbert_novo --is_test
