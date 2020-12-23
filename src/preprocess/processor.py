import os
import re
import json
import logging
from transformers import BertTokenizer
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

ENTITY_TYPES = ['DRUG', 'DRUG_INGREDIENT', 'DISEASE', 'SYMPTOM', 'SYNDROME', 'DISEASE_GROUP',
                'FOOD', 'FOOD_GROUP', 'PERSON_GROUP', 'DRUG_GROUP', 'DRUG_DOSAGE', 'DRUG_TASTE',
                'DRUG_EFFICACY']


class InputExample:
    def __init__(self,
                 set_type,
                 text,
                 labels=None,
                 pseudo=None,
                 distant_labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels
        self.pseudo = pseudo
        self.distant_labels = distant_labels


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class CRFFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None,
                 pseudo=None,
                 distant_labels=None):
        super(CRFFeature, self).__init__(token_ids=token_ids,
                                         attention_masks=attention_masks,
                                         token_type_ids=token_type_ids)
        # labels
        self.labels = labels

        # pseudo
        self.pseudo = pseudo

        # distant labels
        self.distant_labels = distant_labels


class SpanFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 start_ids=None,
                 end_ids=None,
                 pseudo=None):
        super(SpanFeature, self).__init__(token_ids=token_ids,
                                          attention_masks=attention_masks,
                                          token_type_ids=token_type_ids)
        self.start_ids = start_ids
        self.end_ids = end_ids
        # pseudo
        self.pseudo = pseudo

class MRCFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 ent_type=None,
                 start_ids=None,
                 end_ids=None,
                 pseudo=None):
        super(MRCFeature, self).__init__(token_ids=token_ids,
                                         attention_masks=attention_masks,
                                         token_type_ids=token_type_ids)
        self.ent_type = ent_type
        self.start_ids = start_ids
        self.end_ids = end_ids

        # pseudo
        self.pseudo = pseudo


class NERProcessor:
    def __init__(self, cut_sent_len=256):
        self.cut_sent_len = cut_sent_len

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = json.load(f)
        return raw_examples

    @staticmethod
    def _refactor_labels(sent, labels, distant_labels, start_index):
        """
        分句后需要重构 labels 的 offset
        :param sent: 切分并重新合并后的句子
        :param labels: 原始文档级的 labels
        :param distant_labels: 远程监督 label
        :param start_index: 该句子在文档中的起始 offset
        :return (type, entity, offset)
        """
        new_labels, new_distant_labels = [], []
        end_index = start_index + len(sent)

        for _label in labels:
            if start_index <= _label[2] <= _label[3] <= end_index:
                new_offset = _label[2] - start_index

                assert sent[new_offset: new_offset + len(_label[-1])] == _label[-1]

                new_labels.append((_label[1], _label[-1], new_offset))
            # label 被截断的情况
            elif _label[2] < end_index < _label[3]:
                raise RuntimeError(f'{sent}, {_label}')

        for _label in distant_labels:
            if _label in sent:
                new_distant_labels.append(_label)

        return new_labels, new_distant_labels

    def get_examples(self, raw_examples, set_type):
        examples = []

        for i, item in enumerate(raw_examples):
            text = item['text']
            distant_labels = item['candidate_entities']
            pseudo = item['pseudo']

            sentences = cut_sent(text, self.cut_sent_len)
            start_index = 0

            for sent in sentences:
                labels, tmp_distant_labels = self._refactor_labels(sent, item['labels'], distant_labels, start_index)

                start_index += len(sent)

                examples.append(InputExample(set_type=set_type,
                                             text=sent,
                                             labels=labels,
                                             pseudo=pseudo,
                                             distant_labels=tmp_distant_labels))

        return examples


def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens


def cut_sentences_v1(sent):
    """
    the first rank of sentence cut
    """
    sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", sent)  # 单字符断句符
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
    return sent.split("\n")


def cut_sentences_v2(sent):
    """
    the second rank of spilt sentence, split '；' | ';'
    """
    sent = re.sub('([；;])([^”’])', r"\1\n\2", sent)
    return sent.split("\n")


def cut_sent(text, max_seq_len):
    # 将句子分句，细粒度分句后再重新合并
    sentences = []

    # 细粒度划分
    sentences_v1 = cut_sentences_v1(text)
    for sent_v1 in sentences_v1:
        if len(sent_v1) > max_seq_len - 2:
            sentences_v2 = cut_sentences_v2(sent_v1)
            sentences.extend(sentences_v2)
        else:
            sentences.append(sent_v1)

    assert ''.join(sentences) == text

    # 合并
    merged_sentences = []
    start_index_ = 0

    while start_index_ < len(sentences):
        tmp_text = sentences[start_index_]

        end_index_ = start_index_ + 1

        while end_index_ < len(sentences) and \
                len(tmp_text) + len(sentences[end_index_]) <= max_seq_len - 2:
            tmp_text += sentences[end_index_]
            end_index_ += 1

        start_index_ = end_index_

        merged_sentences.append(tmp_text)

    return merged_sentences

def sent_mask(sent, stop_mask_range_list, mask_prob=0.15):
    """
    将句子中的词以 mask prob 的概率随机 mask，
    其中  85% 概率被置为 [mask] 15% 的概率不变。
    :param sent: list of segment words
    :param stop_mask_range_list: 不能 mask 的区域
    :param mask_prob: max mask nums: len(sent) * max_mask_prob
    :return:
    """
    max_mask_token_nums = int(len(sent) * mask_prob)
    mask_nums = 0
    mask_sent = []

    for i in range(len(sent)):

        flag = False
        for _stop_range in stop_mask_range_list:
            if _stop_range[0] <= i <= _stop_range[1]:
                flag = True
                break

        if flag:
            mask_sent.append(sent[i])
            continue

        if mask_nums < max_mask_token_nums:
            # mask_prob 的概率进行 mask, 80% 概率被置为 [mask]，10% 概率被替换， 10% 的概率不变
            if random.random() < mask_prob:
                mask_sent.append('[MASK]')
                mask_nums += 1
            else:
                mask_sent.append(sent[i])
        else:
            mask_sent.append(sent[i])

    return mask_sent


def convert_crf_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                        max_seq_len, ent2id):
    set_type = example.set_type
    raw_text = example.text
    entities = example.labels
    pseudo = example.pseudo

    callback_info = (raw_text,)
    callback_labels = {x: [] for x in ENTITY_TYPES}

    for _label in entities:
        callback_labels[_label[0]].append((_label[1], _label[2]))

    callback_info += (callback_labels,)

    tokens = fine_grade_tokenize(raw_text, tokenizer)
    assert len(tokens) == len(raw_text)

    label_ids = None

    if set_type == 'train':
        # information for dev callback
        label_ids = [0] * len(tokens)

        # tag labels  ent ex. (T1, DRUG_DOSAGE, 447, 450, 小蜜丸)
        for ent in entities:
            ent_type = ent[0]

            ent_start = ent[-1]
            ent_end = ent_start + len(ent[1]) - 1

            if ent_start == ent_end:
                label_ids[ent_start] = ent2id['S-' + ent_type]
            else:
                label_ids[ent_start] = ent2id['B-' + ent_type]
                label_ids[ent_end] = ent2id['E-' + ent_type]
                for i in range(ent_start + 1, ent_end):
                    label_ids[i] = ent2id['I-' + ent_type]

        if len(label_ids) > max_seq_len - 2:
            label_ids = label_ids[:max_seq_len - 2]

        label_ids = [0] + label_ids + [0]

        # pad
        if len(label_ids) < max_seq_len:
            pad_length = max_seq_len - len(label_ids)
            label_ids = label_ids + [0] * pad_length  # CLS SEP PAD label都为O

        assert len(label_ids) == max_seq_len, f'{len(label_ids)}'

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    # if ex_idx < 3:
    #     logger.info(f"*** {set_type}_example-{ex_idx} ***")
    #     logger.info(f'text: {" ".join(tokens)}')
    #     logger.info(f"token_ids: {token_ids}")
    #     logger.info(f"attention_masks: {attention_masks}")
    #     logger.info(f"token_type_ids: {token_type_ids}")
    #     logger.info(f"labels: {label_ids}")

    feature = CRFFeature(
        # bert inputs
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        labels=label_ids,
        pseudo=pseudo
    )

    return feature, callback_info


def convert_span_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                         max_seq_len, ent2id):
    set_type = example.set_type
    raw_text = example.text
    entities = example.labels
    pseudo = example.pseudo

    tokens = fine_grade_tokenize(raw_text, tokenizer)
    assert len(tokens) == len(raw_text)

    callback_labels = {x: [] for x in ENTITY_TYPES}

    for _label in entities:
        callback_labels[_label[0]].append((_label[1], _label[2]))

    callback_info = (raw_text, callback_labels,)

    start_ids, end_ids = None, None

    if set_type == 'train':
        start_ids = [0] * len(tokens)
        end_ids = [0] * len(tokens)

        for _ent in entities:

            ent_type = ent2id[_ent[0]]
            ent_start = _ent[-1]
            ent_end = ent_start + len(_ent[1]) - 1

            start_ids[ent_start] = ent_type
            end_ids[ent_end] = ent_type

        if len(start_ids) > max_seq_len - 2:
            start_ids = start_ids[:max_seq_len - 2]
            end_ids = end_ids[:max_seq_len - 2]

        start_ids = [0] + start_ids + [0]
        end_ids = [0] + end_ids + [0]

        # pad
        if len(start_ids) < max_seq_len:
            pad_length = max_seq_len - len(start_ids)

            start_ids = start_ids + [0] * pad_length  # CLS SEP PAD label都为O
            end_ids = end_ids + [0] * pad_length

        assert len(start_ids) == max_seq_len
        assert len(end_ids) == max_seq_len

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    # if ex_idx < 3:
    #     logger.info(f"*** {set_type}_example-{ex_idx} ***")
    #     logger.info(f'text: {" ".join(tokens)}')
    #     logger.info(f"token_ids: {token_ids}")
    #     logger.info(f"attention_masks: {attention_masks}")
    #     logger.info(f"token_type_ids: {token_type_ids}")
    #     if start_ids and end_ids:
    #         logger.info(f"start_ids: {start_ids}")
    #         logger.info(f"end_ids: {end_ids}")

    feature = SpanFeature(token_ids=token_ids,
                          attention_masks=attention_masks,
                          token_type_ids=token_type_ids,
                          start_ids=start_ids,
                          end_ids=end_ids,
                          pseudo=pseudo)

    return feature, callback_info

def convert_mrc_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                        max_seq_len, ent2id, ent2query, mask_prob=None):
    set_type = example.set_type
    text_b = example.text
    entities = example.labels
    pseudo = example.pseudo

    features = []
    callback_info = []

    tokens_b = fine_grade_tokenize(text_b, tokenizer)
    assert len(tokens_b) == len(text_b)

    label_dict = defaultdict(list)

    for ent in entities:
        ent_type = ent[0]
        ent_start = ent[-1]
        ent_end = ent_start + len(ent[1]) - 1
        label_dict[ent_type].append((ent_start, ent_end, ent[1]))

    # 训练数据中构造
    if set_type == 'train':

        # 每一类为一个 example
        # for _type in label_dict.keys():
        for _type in ENTITY_TYPES:
            start_ids = [0] * len(tokens_b)
            end_ids = [0] * len(tokens_b)

            stop_mask_ranges = []

            text_a = ent2query[_type]
            tokens_a = fine_grade_tokenize(text_a, tokenizer)

            for _label in label_dict[_type]:
                start_ids[_label[0]] = 1
                end_ids[_label[1]] = 1

                stop_mask_ranges.append((_label[0], _label[1]))

            if len(start_ids) > max_seq_len - len(tokens_a) - 3:
                start_ids = start_ids[:max_seq_len - len(tokens_a) - 3]
                end_ids = end_ids[:max_seq_len - len(tokens_a) - 3]
                print('产生了不该有的截断')

            start_ids = [0] + [0] * len(tokens_a) + [0] + start_ids + [0]
            end_ids = [0] + [0] * len(tokens_a) + [0] + end_ids + [0]

            # pad
            if len(start_ids) < max_seq_len:
                pad_length = max_seq_len - len(start_ids)

                start_ids = start_ids + [0] * pad_length  # CLS SEP PAD label都为O
                end_ids = end_ids + [0] * pad_length

            assert len(start_ids) == max_seq_len
            assert len(end_ids) == max_seq_len

            # 随机mask
            if mask_prob:
                tokens_b = sent_mask(tokens_b, stop_mask_ranges, mask_prob=mask_prob)

            encode_dict = tokenizer.encode_plus(text=tokens_a,
                                                text_pair=tokens_b,
                                                max_length=max_seq_len,
                                                pad_to_max_length=True,
                                                truncation_strategy='only_second',
                                                is_pretokenized=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True)

            token_ids = encode_dict['input_ids']
            attention_masks = encode_dict['attention_mask']
            token_type_ids = encode_dict['token_type_ids']

            # if ex_idx < 3:
            #     logger.info(f"*** {set_type}_example-{ex_idx} ***")
            #     logger.info(f'text: {" ".join(tokens_b)}')
            #     logger.info(f"token_ids: {token_ids}")
            #     logger.info(f"attention_masks: {attention_masks}")
            #     logger.info(f"token_type_ids: {token_type_ids}")
            #     logger.info(f'entity type: {_type}')
            #     logger.info(f"start_ids: {start_ids}")
            #     logger.info(f"end_ids: {end_ids}")

            feature = MRCFeature(token_ids=token_ids,
                                 attention_masks=attention_masks,
                                 token_type_ids=token_type_ids,
                                 ent_type=ent2id[_type],
                                 start_ids=start_ids,
                                 end_ids=end_ids,
                                 pseudo=pseudo
                                 )

            features.append(feature)

    # 测试数据构造，为每一类单独构造一个 example
    else:
        for _type in ENTITY_TYPES:
            text_a = ent2query[_type]
            tokens_a = fine_grade_tokenize(text_a, tokenizer)

            encode_dict = tokenizer.encode_plus(text=tokens_a,
                                                text_pair=tokens_b,
                                                max_length=max_seq_len,
                                                pad_to_max_length=True,
                                                truncation_strategy='only_second',
                                                is_pretokenized=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True)

            token_ids = encode_dict['input_ids']
            attention_masks = encode_dict['attention_mask']
            token_type_ids = encode_dict['token_type_ids']

            tmp_callback = (text_b, len(tokens_a) + 2, _type)  # (text, text_offset, type, labels)
            tmp_callback_labels = []

            for _label in label_dict[_type]:
                tmp_callback_labels.append((_label[2], _label[0]))

            tmp_callback += (tmp_callback_labels, )

            callback_info.append(tmp_callback)

            feature = MRCFeature(token_ids=token_ids,
                                 attention_masks=attention_masks,
                                 token_type_ids=token_type_ids,
                                 ent_type=ent2id[_type])

            features.append(feature)

    return features, callback_info



def convert_examples_to_features(task_type, examples, max_seq_len, bert_dir, ent2id):
    assert task_type in ['crf', 'span', 'mrc']

    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))

    features = []

    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')
    type2id = {x: i for i, x in enumerate(ENTITY_TYPES)}

    for i, example in enumerate(examples):
        if task_type == 'crf':
            feature, tmp_callback = convert_crf_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                ent2id=ent2id,
                tokenizer=tokenizer
            )
        elif task_type == 'mrc':
            feature, tmp_callback = convert_mrc_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                ent2id=type2id,
                ent2query=ent2id,
                tokenizer=tokenizer
            )
        else:
            feature, tmp_callback = convert_span_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                ent2id=ent2id,
                tokenizer=tokenizer
            )

        if feature is None:
            continue

        if task_type == 'mrc':
            features.extend(feature)
            callback_info.extend(tmp_callback)
        else:
            features.append(feature)
            callback_info.append(tmp_callback)

    logger.info(f'Build {len(features)} features')

    out = (features, )

    if not len(callback_info):
        return out

    type_weight = {}  # 统计每一类的比例，用于计算 micro-f1
    for _type in ENTITY_TYPES:
        type_weight[_type] = 0.

    count = 0.

    if task_type == 'mrc':
        for _callback in callback_info:
            type_weight[_callback[-2]] += len(_callback[-1])
            count += len(_callback[-1])
    else:
        for _callback in callback_info:
            for _type in _callback[1]:
                type_weight[_type] += len(_callback[1][_type])
                count += len(_callback[1][_type])

    for key in type_weight:
        type_weight[key] /= count

    out += ((callback_info, type_weight), )

    return out


if __name__ == '__main__':
    pass
