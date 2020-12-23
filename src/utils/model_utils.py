import os
import math
import torch
import torch.nn as nn
from torchcrf import CRF
from itertools import repeat
from transformers import BertModel
from src.utils.functions_utils import vote
from src.utils.evaluator import crf_decode, span_decode


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()


        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_pred, target,
                                                                                   reduction=self.reduction,
                                                                                   ignore_index=self.ignore_index)

class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss

class SpatialDropout(nn.Module):
    """
    对字级别的向量进行丢弃
    """
    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), *repeat(1, input.dim() - 2), input.size(2))

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 cond_shape,
                 eps=1e-12):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))

        self.weight_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(cond_shape, normalized_shape, bias=False)

        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)

        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)

        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)

        outputs = outputs / std  # (b, s, h)

        outputs = outputs * weight + bias

        return outputs

class BaseModel(nn.Module):
    def __init__(self,
                 bert_dir,
                 dropout_prob):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')

        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir,
                                                     output_hidden_states=True,
                                                     hidden_dropout_prob=dropout_prob)

        self.bert_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


# baseline
class CRFModel(BaseModel):
    def __init__(self,
                 bert_dir,
                 num_tags,
                 dropout_prob=0.1,
                 **kwargs):
        super(CRFModel, self).__init__(bert_dir=bert_dir, dropout_prob=dropout_prob)

        out_dims = self.bert_config.hidden_size

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        out_dims = mid_linear_dims

        self.classifier = nn.Linear(out_dims, num_tags)

        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.2)

        self.crf_module = CRF(num_tags=num_tags, batch_first=True)

        init_blocks = [self.mid_linear, self.classifier]

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                labels=None,
                pseudo=None):

        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        # 常规
        seq_out = bert_outputs[0]

        seq_out = self.mid_linear(seq_out)

        emissions = self.classifier(seq_out)

        if labels is not None:
            if pseudo is not None:
                # (batch,)
                tokens_loss = -1. * self.crf_module(emissions=emissions,
                                                    tags=labels.long(),
                                                    mask=attention_masks.byte(),
                                                    reduction='none')

                # nums of pseudo data
                pseudo_nums = pseudo.sum().item()
                total_nums = token_ids.shape[0]

                # learning parameter
                rate = torch.sigmoid(self.loss_weight)
                if pseudo_nums == 0:
                    loss_0 = tokens_loss.mean()
                    loss_1 = (rate*pseudo*tokens_loss).sum()
                else:
                    if total_nums == pseudo_nums:
                        loss_0 = 0
                    else:
                        loss_0 = ((1 - rate) * (1 - pseudo) * tokens_loss).sum() / (total_nums - pseudo_nums)
                    loss_1 = (rate*pseudo*tokens_loss).sum() / pseudo_nums

                tokens_loss = loss_0 + loss_1

            else:
                tokens_loss = -1. * self.crf_module(emissions=emissions,
                                                    tags=labels.long(),
                                                    mask=attention_masks.byte(),
                                                    reduction='mean')

            out = (tokens_loss,)

        else:
            tokens_out = self.crf_module.decode(emissions=emissions, mask=attention_masks.byte())

            out = (tokens_out, emissions)

        return out


class SpanModel(BaseModel):
    def __init__(self,
                 bert_dir,
                 num_tags,
                 dropout_prob=0.1,
                 loss_type='ce',
                 **kwargs):
        """
        tag the subject and object corresponding to the predicate
        :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
        """
        super(SpanModel, self).__init__(bert_dir, dropout_prob=dropout_prob)

        out_dims = self.bert_config.hidden_size

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.num_tags = num_tags

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        out_dims = mid_linear_dims

        self.start_fc = nn.Linear(out_dims, num_tags)
        self.end_fc = nn.Linear(out_dims, num_tags)

        reduction = 'none'
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        else:
            self.criterion = FocalLoss(reduction=reduction)

        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.2)

        init_blocks = [self.mid_linear, self.start_fc, self.end_fc]

        self._init_weights(init_blocks)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                start_ids=None,
                end_ids=None,
                pseudo=None):

        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        seq_out = bert_outputs[0]

        seq_out = self.mid_linear(seq_out)

        start_logits = self.start_fc(seq_out)
        end_logits = self.end_fc(seq_out)

        out = (start_logits, end_logits, )

        if start_ids is not None and end_ids is not None and self.training:

            start_logits = start_logits.view(-1, self.num_tags)
            end_logits = end_logits.view(-1, self.num_tags)

            # 去掉 padding 部分的标签，计算真实 loss
            active_loss = attention_masks.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]


            if pseudo is not None:
                # (batch,)
                start_loss = self.criterion(start_logits, start_ids.view(-1)).view(-1, 512).mean(dim=-1)
                end_loss = self.criterion(end_logits, end_ids.view(-1)).view(-1, 512).mean(dim=-1)

                # nums of pseudo data
                pseudo_nums = pseudo.sum().item()
                total_nums = token_ids.shape[0]

                # learning parameter
                rate = torch.sigmoid(self.loss_weight)
                if pseudo_nums == 0:
                    start_loss = start_loss.mean()
                    end_loss = end_loss.mean()
                else:
                    if total_nums == pseudo_nums:
                        start_loss = (rate*pseudo*start_loss).sum() / pseudo_nums
                        end_loss = (rate*pseudo*end_loss).sum() / pseudo_nums
                    else:
                        start_loss = (rate*pseudo*start_loss).sum() / pseudo_nums \
                                     + ((1 - rate) * (1 - pseudo) * start_loss).sum() / (total_nums - pseudo_nums)
                        end_loss = (rate*pseudo*end_loss).sum() / pseudo_nums \
                                     + ((1 - rate) * (1 - pseudo) * end_loss).sum() / (total_nums - pseudo_nums)
            else:
                start_loss = self.criterion(active_start_logits, active_start_labels)
                end_loss = self.criterion(active_end_logits, active_end_labels)

            loss = start_loss + end_loss

            out = (loss, ) + out

        return out

class MRCModel(BaseModel):
    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1,
                 use_type_embed=False,
                 loss_type='ce',
                 **kwargs):
        """
        tag the subject and object corresponding to the predicate
        :param use_type_embed: type embedding for the sentence
        :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
        """
        super(MRCModel, self).__init__(bert_dir, dropout_prob=dropout_prob)

        self.use_type_embed = use_type_embed
        self.use_smooth = loss_type

        out_dims = self.bert_config.hidden_size

        if self.use_type_embed:
            embed_dims = kwargs.pop('predicate_embed_dims', self.bert_config.hidden_size)
            self.type_embedding = nn.Embedding(13, embed_dims)

            self.conditional_layer_norm = ConditionalLayerNorm(out_dims, embed_dims,
                                                               eps=self.bert_config.layer_norm_eps)

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        out_dims = mid_linear_dims

        self.start_fc = nn.Linear(out_dims, 2)
        self.end_fc = nn.Linear(out_dims, 2)

        reduction = 'none'
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        else:
            self.criterion = FocalLoss(reduction=reduction)

        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.2)

        init_blocks = [self.mid_linear, self.start_fc, self.end_fc]

        if self.use_type_embed:
            init_blocks.append(self.type_embedding)

        self._init_weights(init_blocks)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                ent_type=None,
                start_ids=None,
                end_ids=None,
                pseudo=None):

        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        seq_out = bert_outputs[0]

        if self.use_type_embed:
            assert ent_type is not None, \
                'Using predicate embedding, predicate should be implemented'

            predicate_feature = self.type_embedding(ent_type)
            seq_out = self.conditional_layer_norm(seq_out, predicate_feature)

        seq_out = self.mid_linear(seq_out)

        start_logits = self.start_fc(seq_out)
        end_logits = self.end_fc(seq_out)

        out = (start_logits, end_logits, )

        if start_ids is not None and end_ids is not None:
            start_logits = start_logits.view(-1, 2)
            end_logits = end_logits.view(-1, 2)

            # 去掉 text_a 和 padding 部分的标签，计算真实 loss
            active_loss = token_type_ids.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]

            if pseudo is not None:
                # (batch,)
                start_loss = self.criterion(start_logits, start_ids.view(-1)).view(-1, 512).mean(dim=-1)
                end_loss = self.criterion(end_logits, end_ids.view(-1)).view(-1, 512).mean(dim=-1)

                # nums of pseudo data
                pseudo_nums = pseudo.sum().item()
                total_nums = token_ids.shape[0]

                # learning parameter
                rate = torch.sigmoid(self.loss_weight)
                if pseudo_nums == 0:
                    start_loss = start_loss.mean()
                    end_loss = end_loss.mean()
                else:
                    if total_nums == pseudo_nums:
                        start_loss = (rate*pseudo*start_loss).sum() / pseudo_nums
                        end_loss = (rate*pseudo*end_loss).sum() / pseudo_nums
                    else:
                        start_loss = (rate*pseudo*start_loss).sum() / pseudo_nums \
                                     + ((1 - rate) * (1 - pseudo) * start_loss).sum() / (total_nums - pseudo_nums)
                        end_loss = (rate*pseudo*end_loss).sum() / pseudo_nums \
                                     + ((1 - rate) * (1 - pseudo) * end_loss).sum() / (total_nums - pseudo_nums)
            else:
                start_loss = self.criterion(active_start_logits, active_start_labels)
                end_loss = self.criterion(active_end_logits, active_end_labels)

            loss = start_loss + end_loss

            out = (loss, ) + out

        return out


class EnsembleCRFModel:
    def __init__(self, model_path_list, bert_dir_list, num_tags, device, lamb=1/3):

        self.models = []
        self.crf_module = CRF(num_tags=num_tags, batch_first=True)
        self.lamb = lamb

        for idx, _path in enumerate(model_path_list):
            print(f'Load model from {_path}')
            

            print(f'Load model type: {bert_dir_list[0]}')
            model = CRFModel(bert_dir=bert_dir_list[0], num_tags=num_tags)

            
            model.load_state_dict(torch.load(_path, map_location=torch.device('cpu')))

            model.eval()
            model.to(device)

            self.models.append(model)
            if idx == 0:
                print(f'Load CRF weight from {_path}')
                self.crf_module.load_state_dict(model.crf_module.state_dict())
                self.crf_module.to(device)

    def weight(self, t):
        """
        牛顿冷却定律加权融合
        """
        return math.exp(-self.lamb*t)

    def predict(self, model_inputs):
        weight_sum = 0.
        logits = None
        attention_masks = model_inputs['attention_masks']
        
        for idx, model in enumerate(self.models):
            # 使用牛顿冷却概率融合
            weight = self.weight(idx)

            # 使用概率平均融合
            # weight = 1 / len(self.models)

            tmp_logits = model(**model_inputs)[1] * weight
            weight_sum += weight
            
            if logits is None:
                logits = tmp_logits
            else:
                logits += tmp_logits

        logits = logits / weight_sum

        tokens_out = self.crf_module.decode(emissions=logits, mask=attention_masks.byte())

        return tokens_out

    def vote_entities(self, model_inputs, sent, id2ent, threshold):
        entities_ls = []
        for idx, model in enumerate(self.models):
            tmp_tokens = model(**model_inputs)[0][0]
            tmp_entities = crf_decode(tmp_tokens, sent, id2ent)
            entities_ls.append(tmp_entities)

        return vote(entities_ls, threshold)


class EnsembleSpanModel:
    def __init__(self, model_path_list, bert_dir_list, num_tags, device):

        self.models = []

        for idx, _path in enumerate(model_path_list):
            print(f'Load model from {_path}')

            print(f'Load model type: {bert_dir_list[0]}')
            model = SpanModel(bert_dir=bert_dir_list[0], num_tags=num_tags)

            model.load_state_dict(torch.load(_path, map_location=torch.device('cpu')))

            model.eval()
            model.to(device)

            self.models.append(model)

    def predict(self, model_inputs):
        start_logits, end_logits = None, None

        for idx, model in enumerate(self.models):

            # 使用概率平均融合
            weight = 1 / len(self.models)

            tmp_start_logits, tmp_end_logits = model(**model_inputs)

            tmp_start_logits = tmp_start_logits * weight
            tmp_end_logits = tmp_end_logits * weight

            if start_logits is None:
                start_logits = tmp_start_logits
                end_logits = tmp_end_logits
            else:
                start_logits += tmp_start_logits
                end_logits += tmp_end_logits

        return start_logits, end_logits

    def vote_entities(self, model_inputs, sent, id2ent, threshold):
        entities_ls = []

        for idx, model in enumerate(self.models):

            start_logits, end_logits = model(**model_inputs)
            start_logits = start_logits[0].cpu().numpy()[1:1 + len(sent)]
            end_logits = end_logits[0].cpu().numpy()[1:1 + len(sent)]

            decode_entities = span_decode(start_logits, end_logits, sent, id2ent)

            entities_ls.append(decode_entities)

        return vote(entities_ls, threshold)


def build_model(task_type, bert_dir, **kwargs):
    assert task_type in ['crf', 'span', 'mrc']

    if task_type == 'crf':
        model = CRFModel(bert_dir=bert_dir,
                         num_tags=kwargs.pop('num_tags'),
                         dropout_prob=kwargs.pop('dropout_prob', 0.1))

    elif task_type == 'mrc':
        model = MRCModel(bert_dir=bert_dir,
                         dropout_prob=kwargs.pop('dropout_prob', 0.1),
                         use_type_embed=kwargs.pop('use_type_embed'),
                         loss_type=kwargs.pop('loss_type', 'ce'))

    else:
        model = SpanModel(bert_dir=bert_dir,
                          num_tags=kwargs.pop('num_tags'),
                          dropout_prob=kwargs.pop('dropout_prob', 0.1),
                          loss_type=kwargs.pop('loss_type', 'ce'))

    return model
