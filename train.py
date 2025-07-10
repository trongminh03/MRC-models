# train_data = 'drive/MyDrive/CODE/ViSpokenSQuAD/dataset/train.json'
# dev_data = 'drive/MyDrive/CODE/ViSpokenSQuAD/dataset/dev.json'
# test_data = 'drive/MyDrive/CODE/ViSpokenSQuAD/dataset/test.json'
import argparse
import json
from question_answering import QuestionAnsweringModel, QuestionAnsweringArgs


def convert_data(data):
    converted = []
    for d in data:
        t = {}
        t['context'] = d['paragraphs'][0]['context']
        qas = []
        for q in d['paragraphs'][0]['qas']:
            q['is_impossible'] = False
            qas.append(q)
        t['qas'] = qas
        converted.append(t)
    return converted


def read_data(PATH):
    with open(PATH+'train.json', 'r', encoding='utf8') as f:
        TRAIN = json.load(f)

    with open(PATH+'dev.json', 'r', encoding='utf8') as f:
        DEV = json.load(f)

    # with open(PATH+'dev.json', 'r', encoding='utf8') as f:
    #     TEST = json.load(f)

    train_squad = convert_data(TRAIN['data'])
    dev_squad = convert_data(DEV['data'])
    # test_squad = convert_data(TEST['data'])

    # return train_squad, dev_squad, test_squad
    return train_squad, dev_squad


def build_model(args=None):
    model_args = QuestionAnsweringArgs()
    model_args.max_seq_length = args.max_seq_length
    # model_args.evaluate_during_training = True
    model_args.num_train_epochs = args.epochs
    model_args.train_batch_size = args.batch_size
    model_args.use_early_stopping = args.early_stopping
    model_args.use_multiprocessing = args.multiprocessing
    model_args.use_multiprocessing_for_evaluation = args.multiprocessing_evaluation
    model_args.cache_dir = args.output_path + '/cache'
    model_args.overwrite_output_dir = True
    model_args.output_dir = args.output_path
    model_args.config = {
      "bos_token": "[CLS]",
      "cls_token": "[CLS]",
      "do_lower_case": False,
      "eos_token": "[SEP]",
      "mask_token": "[MASK]",
      "name_or_path": "microsoft/deberta-v3-xsmall",
      "pad_token": "[PAD]",
      "sep_token": "[SEP]",
      "sp_model_kwargs": {},
      "special_tokens_map_file": None,
      "split_by_punct": False,
      "tokenizer_class": "DebertaV2Tokenizer",
      "unk_token": "[UNK]",
      "vocab_type": "spm"
    }
    
    model = QuestionAnsweringModel(
        args.type, args.model,
        cuda_device=args.device,
        args=model_args,
        ignore_mismatched_sizes=True,
    )
    return model


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/data")
    parser.add_argument('--type', type=str, default="auto")
    parser.add_argument('--model', type=str, default="auto")
    parser.add_argument('--output_path', type=str, default="/home/data")
    parser.add_argument('--device', type=int, default=0,)
    
     # Additional model arguments
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--multiprocessing', action='store_true')
    parser.add_argument('--multiprocessing_evaluation', action='store_true')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    train, val = read_data(args.path)
    # import IPython; IPython.embed()
    model = build_model(args=args)

    model.train_model(train)
