import torch
import logging
import numpy as np
from collections import defaultdict
from src.preprocess.processor import ENTITY_TYPES

logger = logging.getLogger(__name__)


def get_base_out(model, loader, device):
    """
    每一个任务的 forward 都一样，封装起来
    """
    model.eval()

    with torch.no_grad():
        for idx, _batch in enumerate(loader):

            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)

            tmp_out = model(**_batch)

            yield tmp_out


def crf_decode(decode_tokens, raw_text, id2ent):
    """
    CRF 解码，用于解码 time loc 的提取
    """
    predict_entities = {}

    decode_tokens = decode_tokens[1:-1]  # 除去 CLS SEP token

    index_ = 0

    while index_ < len(decode_tokens):

        token_label = id2ent[decode_tokens[index_]].split('-')

        if token_label[0].startswith('S'):
            token_type = token_label[1]
            tmp_ent = raw_text[index_]

            if token_type not in predict_entities:
                predict_entities[token_type] = [(tmp_ent, index_)]
            else:
                predict_entities[token_type].append((tmp_ent, int(index_)))

            index_ += 1

        elif token_label[0].startswith('B'):
            token_type = token_label[1]
            start_index = index_

            index_ += 1
            while index_ < len(decode_tokens):
                temp_token_label = id2ent[decode_tokens[index_]].split('-')

                if temp_token_label[0].startswith('I') and token_type == temp_token_label[1]:
                    index_ += 1
                elif temp_token_label[0].startswith('E') and token_type == temp_token_label[1]:
                    end_index = index_
                    index_ += 1

                    tmp_ent = raw_text[start_index: end_index + 1]

                    if token_type not in predict_entities:
                        predict_entities[token_type] = [(tmp_ent, start_index)]
                    else:
                        predict_entities[token_type].append((tmp_ent, int(start_index)))

                    break
                else:
                    break
        else:
            index_ += 1

    return predict_entities


# 严格解码 baseline
def span_decode(start_logits, end_logits, raw_text, id2ent):
    predict_entities = defaultdict(list)

    start_pred = np.argmax(start_logits, -1)
    end_pred = np.argmax(end_logits, -1)

    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                tmp_ent = raw_text[i:i + j + 1]
                predict_entities[id2ent[s_type]].append((tmp_ent, i))
                break

    return predict_entities

# 严格解码 baseline
def mrc_decode(start_logits, end_logits, raw_text):
    predict_entities = []
    start_pred = np.argmax(start_logits, -1)
    end_pred = np.argmax(end_logits, -1)

    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                tmp_ent = raw_text[i:i+j+1]
                predict_entities.append((tmp_ent, i))
                break

    return predict_entities


def calculate_metric(gt, predict):
    """
    计算 tp fp fn
    """
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gt:
            if entity_predict[0] == entity_gt[0] and entity_predict[1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1

    fn = len(gt) - tp

    return np.array([tp, fp, fn])


def get_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])


def crf_evaluation(model, dev_info, device, ent2id):
    dev_loader, (dev_callback_info, type_weight) = dev_info

    pred_tokens = []

    for tmp_pred in get_base_out(model, dev_loader, device):
        pred_tokens.extend(tmp_pred[0])

    assert len(pred_tokens) == len(dev_callback_info)

    id2ent = {ent2id[key]: key for key in ent2id.keys()}

    role_metric = np.zeros([13, 3])

    mirco_metrics = np.zeros(3)

    for tmp_tokens, tmp_callback in zip(pred_tokens, dev_callback_info):

        text, gt_entities = tmp_callback

        tmp_metric = np.zeros([13, 3])

        pred_entities = crf_decode(tmp_tokens, text, id2ent)

        for idx, _type in enumerate(ENTITY_TYPES):
            if _type not in pred_entities:
                pred_entities[_type] = []

            tmp_metric[idx] += calculate_metric(gt_entities[_type], pred_entities[_type])

        role_metric += tmp_metric

    for idx, _type in enumerate(ENTITY_TYPES):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1], role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_type]

    metric_str = f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                 f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2]


def span_evaluation(model, dev_info, device, ent2id):
    dev_loader, (dev_callback_info, type_weight) = dev_info

    start_logits, end_logits = None, None

    model.eval()

    for tmp_pred in get_base_out(model, dev_loader, device):
        tmp_start_logits = tmp_pred[0].cpu().numpy()
        tmp_end_logits = tmp_pred[1].cpu().numpy()

        if start_logits is None:
            start_logits = tmp_start_logits
            end_logits = tmp_end_logits
        else:
            start_logits = np.append(start_logits, tmp_start_logits, axis=0)
            end_logits = np.append(end_logits, tmp_end_logits, axis=0)

    assert len(start_logits) == len(end_logits) == len(dev_callback_info)

    role_metric = np.zeros([13, 3])

    mirco_metrics = np.zeros(3)

    id2ent = {ent2id[key]: key for key in ent2id.keys()}

    for tmp_start_logits, tmp_end_logits, tmp_callback \
            in zip(start_logits, end_logits, dev_callback_info):

        text, gt_entities = tmp_callback

        tmp_start_logits = tmp_start_logits[1:1 + len(text)]
        tmp_end_logits = tmp_end_logits[1:1 + len(text)]

        pred_entities = span_decode(tmp_start_logits, tmp_end_logits, text, id2ent)

        for idx, _type in enumerate(ENTITY_TYPES):
            if _type not in pred_entities:
                pred_entities[_type] = []

            role_metric[idx] += calculate_metric(gt_entities[_type], pred_entities[_type])

    for idx, _type in enumerate(ENTITY_TYPES):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1], role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_type]

    metric_str = f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                 f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2]

def mrc_evaluation(model, dev_info, device):
    dev_loader, (dev_callback_info, type_weight) = dev_info

    start_logits, end_logits = None, None

    model.eval()

    for tmp_pred in get_base_out(model, dev_loader, device):
        tmp_start_logits = tmp_pred[0].cpu().numpy()
        tmp_end_logits = tmp_pred[1].cpu().numpy()

        if start_logits is None:
            start_logits = tmp_start_logits
            end_logits = tmp_end_logits
        else:
            start_logits = np.append(start_logits, tmp_start_logits, axis=0)
            end_logits = np.append(end_logits, tmp_end_logits, axis=0)

    assert len(start_logits) == len(end_logits) == len(dev_callback_info)

    role_metric = np.zeros([13, 3])

    mirco_metrics = np.zeros(3)

    id2ent = {x: i for i, x in enumerate(ENTITY_TYPES)}

    for tmp_start_logits, tmp_end_logits, tmp_callback \
            in zip(start_logits, end_logits, dev_callback_info):

        text, text_offset, ent_type, gt_entities = tmp_callback

        tmp_start_logits = tmp_start_logits[text_offset:text_offset+len(text)]
        tmp_end_logits = tmp_end_logits[text_offset:text_offset+len(text)]

        pred_entities = mrc_decode(tmp_start_logits, tmp_end_logits, text)

        role_metric[id2ent[ent_type]] += calculate_metric(gt_entities, pred_entities)

    for idx, _type in enumerate(ENTITY_TYPES):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1], role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_type]

    metric_str = f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                  f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2]
