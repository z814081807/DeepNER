import os
import json
import torch
from collections import defaultdict
from transformers import BertTokenizer
from src.utils.model_utils import CRFModel, SpanModel, EnsembleCRFModel, EnsembleSpanModel
from src.utils.evaluator import crf_decode, span_decode
from src.utils.functions_utils import load_model_and_parallel, ensemble_vote
from src.preprocess.processor import cut_sent, fine_grade_tokenize

MID_DATA_DIR = "./data/mid_data"
RAW_DATA_DIR = "./data/raw_data_random"
SUBMIT_DIR = "./result"
GPU_IDS = "0"

LAMBDA = 0.3
THRESHOLD = 0.9
MAX_SEQ_LEN = 512

TASK_TYPE = "crf"  # choose crf or span
VOTE = True  # choose True or False
VERSION = "mixed"  # choose single or ensemble or mixed ; if mixed  VOTE and TAST_TYPE is useless.

# single_predict
BERT_TYPE = "uer_large"  # roberta_wwm / ernie_1 / uer_large

BERT_DIR = f"./bert/torch_{BERT_TYPE}"
with open('./best_ckpt_path.txt', 'r', encoding='utf-8') as f:
    CKPT_PATH = f.read().strip()

# ensemble_predict
BERT_DIR_LIST = ["./bert/torch_uer_large", "./bert/torch_roberta_wwm"]

with open('./best_ckpt_path.txt', 'r', encoding='utf-8') as f:
    ENSEMBLE_DIR_LIST = f.readlines()
    print('ENSEMBLE_DIR_LIST:{}'.format(ENSEMBLE_DIR_LIST))


# mixed_predict
MIX_BERT_DIR = "./bert/torch_uer_large"

with open('./best_ckpt_path.txt', 'r', encoding='utf-8') as f:
    MIX_DIR_LIST = f.readlines()
    print('MIX_DIR_LIST:{}'.format(MIX_DIR_LIST))


def prepare_info():
    info_dict = {}
    with open(os.path.join(MID_DATA_DIR, f'{TASK_TYPE}_ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)

    with open(os.path.join(RAW_DATA_DIR, 'test.json'), encoding='utf-8') as f:
        info_dict['examples'] = json.load(f)

    info_dict['id2ent'] = {ent2id[key]: key for key in ent2id.keys()}

    info_dict['tokenizer'] = BertTokenizer(os.path.join(BERT_DIR, 'vocab.txt'))

    return info_dict


def mixed_prepare_info(mixed='crf'):
    info_dict = {}
    with open(os.path.join(MID_DATA_DIR, f'{mixed}_ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)

    with open(os.path.join(RAW_DATA_DIR, 'test.json'), encoding='utf-8') as f:
        info_dict['examples'] = json.load(f)

    info_dict['id2ent'] = {ent2id[key]: key for key in ent2id.keys()}

    info_dict['tokenizer'] = BertTokenizer(os.path.join(BERT_DIR, 'vocab.txt'))

    return info_dict


def base_predict(model, device, info_dict, ensemble=False, mixed=''):
    labels = defaultdict(list)

    tokenizer = info_dict['tokenizer']
    id2ent = info_dict['id2ent']

    with torch.no_grad():
        for _ex in info_dict['examples']:
            ex_idx = _ex['id']
            raw_text = _ex['text']

            if not len(raw_text):
                labels[ex_idx] = []
                print('{}为空'.format(ex_idx))
                continue

            sentences = cut_sent(raw_text, MAX_SEQ_LEN)

            start_index = 0

            for sent in sentences:

                sent_tokens = fine_grade_tokenize(sent, tokenizer)

                encode_dict = tokenizer.encode_plus(text=sent_tokens,
                                                    max_length=MAX_SEQ_LEN,
                                                    is_pretokenized=True,
                                                    pad_to_max_length=False,
                                                    return_tensors='pt',
                                                    return_token_type_ids=True,
                                                    return_attention_mask=True)

                model_inputs = {'token_ids': encode_dict['input_ids'],
                                'attention_masks': encode_dict['attention_mask'],
                                'token_type_ids': encode_dict['token_type_ids']}

                for key in model_inputs:
                    model_inputs[key] = model_inputs[key].to(device)

                if ensemble:
                    if TASK_TYPE == 'crf':
                        if VOTE:
                            decode_entities = model.vote_entities(model_inputs, sent, id2ent, THRESHOLD)
                        else:
                            pred_tokens = model.predict(model_inputs)[0]
                            decode_entities = crf_decode(pred_tokens, sent, id2ent)
                    else:
                        if VOTE:
                            decode_entities = model.vote_entities(model_inputs, sent, id2ent, THRESHOLD)
                        else:
                            start_logits, end_logits = model.predict(model_inputs)
                            start_logits = start_logits[0].cpu().numpy()[1:1 + len(sent)]
                            end_logits = end_logits[0].cpu().numpy()[1:1 + len(sent)]

                            decode_entities = span_decode(start_logits, end_logits, sent, id2ent)

                else:

                    if mixed:
                        if mixed == 'crf':
                            pred_tokens = model(**model_inputs)[0][0]
                            decode_entities = crf_decode(pred_tokens, sent, id2ent)
                        else:
                            start_logits, end_logits = model(**model_inputs)

                            start_logits = start_logits[0].cpu().numpy()[1:1 + len(sent)]
                            end_logits = end_logits[0].cpu().numpy()[1:1 + len(sent)]

                            decode_entities = span_decode(start_logits, end_logits, sent, id2ent)

                    else:
                        if TASK_TYPE == 'crf':
                            pred_tokens = model(**model_inputs)[0][0]
                            decode_entities = crf_decode(pred_tokens, sent, id2ent)
                        else:
                            start_logits, end_logits = model(**model_inputs)

                            start_logits = start_logits[0].cpu().numpy()[1:1+len(sent)]
                            end_logits = end_logits[0].cpu().numpy()[1:1+len(sent)]

                            decode_entities = span_decode(start_logits, end_logits, sent, id2ent)


                for _ent_type in decode_entities:
                    for _ent in decode_entities[_ent_type]:
                        tmp_start = _ent[1] + start_index
                        tmp_end = tmp_start + len(_ent[0])

                        assert raw_text[tmp_start: tmp_end] == _ent[0]

                        labels[ex_idx].append((_ent_type, tmp_start, tmp_end, _ent[0]))

                start_index += len(sent)

                if not len(labels[ex_idx]):
                    labels[ex_idx] = []

    return labels


def single_predict():
    save_dir = os.path.join(SUBMIT_DIR, VERSION)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    info_dict = prepare_info()

    if TASK_TYPE == 'crf':
        model = CRFModel(bert_dir=BERT_DIR, num_tags=len(info_dict['id2ent']))
    else:
        model = SpanModel(bert_dir=BERT_DIR, num_tags=len(info_dict['id2ent'])+1)

    print(f'Load model from {CKPT_PATH}')
    model, device = load_model_and_parallel(model, GPU_IDS, CKPT_PATH)
    model.eval()

    labels = base_predict(model, device, info_dict)

    for key in labels.keys():
        with open(os.path.join(save_dir, f'{key}.ann'), 'w', encoding='utf-8') as f:
            if not len(labels[key]):
                print(key)
                f.write("")
            else:
                for idx, _label in enumerate(labels[key]):
                    f.write(f'T{idx + 1}\t{_label[0]} {_label[1]} {_label[2]}\t{_label[3]}\n')


def ensemble_predict():
    save_dir = os.path.join(SUBMIT_DIR, VERSION)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    info_dict = prepare_info()

    model_path_list = [x.strip() for x in ENSEMBLE_DIR_LIST]
    print('model_path_list:{}'.format(model_path_list))

    device = torch.device(f'cuda:{GPU_IDS[0]}')


    if TASK_TYPE == 'crf':
        model = EnsembleCRFModel(model_path_list=model_path_list,
                                 bert_dir_list=BERT_DIR_LIST,
                                 num_tags=len(info_dict['id2ent']),
                                 device=device,
                                 lamb=LAMBDA)
    else:
        model = EnsembleSpanModel(model_path_list=model_path_list,
                                 bert_dir_list=BERT_DIR_LIST,
                                 num_tags=len(info_dict['id2ent'])+1,
                                 device=device)


    labels = base_predict(model, device, info_dict, ensemble=True)
    

    for key in labels.keys():
        with open(os.path.join(save_dir, f'{key}.ann'), 'w', encoding='utf-8') as f:
            if not len(labels[key]):
                print(key)
                f.write("")
            else:
                for idx, _label in enumerate(labels[key]):
                    f.write(f'T{idx + 1}\t{_label[0]} {_label[1]} {_label[2]}\t{_label[3]}\n')

def mixed_predict():
    save_dir = os.path.join(SUBMIT_DIR, VERSION)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    model_path_list = [x.strip() for x in MIX_DIR_LIST]
    print('model_path_list:{}'.format(model_path_list))

    all_labels = []

    for i, model_path in enumerate(model_path_list):
        if i <= 4:
            info_dict = mixed_prepare_info(mixed='span')

            model = SpanModel(bert_dir=MIX_BERT_DIR, num_tags=len(info_dict['id2ent']) + 1)
            print(f'Load model from {model_path}')
            model, device = load_model_and_parallel(model, GPU_IDS, model_path)
            model.eval()
            labels = base_predict(model, device, info_dict, ensemble=False, mixed='span')

        else:
            info_dict = mixed_prepare_info(mixed='crf')

            model = CRFModel(bert_dir=MIX_BERT_DIR, num_tags=len(info_dict['id2ent']))
            print(f'Load model from {model_path}')
            model, device = load_model_and_parallel(model, GPU_IDS, model_path)
            model.eval()
            labels = base_predict(model, device, info_dict, ensemble=False, mixed='crf')

        all_labels.append(labels)

    labels = ensemble_vote(all_labels, THRESHOLD)

    # for key in labels.keys():

    for key in range(1500, 1997):
        with open(os.path.join(save_dir, f'{key}.ann'), 'w', encoding='utf-8') as f:
            if not len(labels[key]):
                print(key)
                f.write("")
            else:
                for idx, _label in enumerate(labels[key]):
                    f.write(f'T{idx + 1}\t{_label[0]} {_label[1]} {_label[2]}\t{_label[3]}\n')



if __name__ == '__main__':
    assert VERSION in ['single', 'ensemble', 'mixed'], 'VERSION mismatch'

    if VERSION == 'single':
        single_predict()
    elif VERSION == 'ensemble':
        if VOTE:
            print("————————开始投票：————————")
        ensemble_predict()

    elif VERSION == 'mixed':
        print("————————开始混合投票：————————")
        mixed_predict()

    # 压缩result.zip
    import zipfile

    def zip_file(src_dir):
        zip_name = src_dir + '.zip'
        z = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
        for dirpath, dirnames, filenames in os.walk(src_dir):
            fpath = dirpath.replace(src_dir, '')
            fpath = fpath and fpath + os.sep or ''
            for filename in filenames:
                z.write(os.path.join(dirpath, filename), fpath + filename)
        print('==压缩成功==')
        z.close()

    zip_file('./result')

