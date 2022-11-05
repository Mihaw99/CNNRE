#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.7.5

import numpy as np
import mindspore
import mindspore.ops as ops


def semeval_scorer(predict_label, true_label, class_num=10):
    import math
    assert true_label.shape[0] == predict_label.shape[0]
    confusion_matrix = np.zeros(shape=[class_num, class_num], dtype=np.float32)
    xDIRx = np.zeros(shape=[class_num], dtype=np.float32)
#     print("true_label.shape[0]: ", true_label.shape[0])
    for i in range(true_label.shape[0]):
        true_idx = math.ceil(true_label[i]/2)
        predict_idx = math.ceil(predict_label[i]/2)
#         print("predict_idx: ", predict_idx)
#         print("true_idx: ", true_idx)
        if true_label[i] == predict_label[i]:
            confusion_matrix[predict_idx][true_idx] += 1
        else:
            if true_idx == predict_idx:
                xDIRx[predict_idx] += 1
            else:
                confusion_matrix[predict_idx][true_idx] += 1

    col_sum = np.sum(confusion_matrix, axis=0).reshape(-1)
    row_sum = np.sum(confusion_matrix, axis=1).reshape(-1)
    f1 = np.zeros(shape=[class_num], dtype=np.float32)

    for i in range(0, class_num):  # ignore the 'Other'
        try:
            p = float(confusion_matrix[i][i]) / float(col_sum[i] + xDIRx[i])
            r = float(confusion_matrix[i][i]) / float(row_sum[i] + xDIRx[i])
            f1[i] = (2 * p * r / (p + r))
        except:
            pass
    actual_class = 0
    total_f1 = 0.0
    for i in range(1, class_num):
        if f1[i] > 0.0:  # classes that not in the predict label are not considered
            actual_class += 1
            total_f1 += f1[i]
    try:
        macro_f1 = total_f1 / actual_class
    except:
        macro_f1 = 0.0
    return macro_f1


class Eval(object):
    def __init__(self, config):
        pass

    def evaluate(self, model, criterion, data_loader):
        predict_label = []
        true_label = []
        total_loss = 0.0
        # with ops.stop_gradient():
        # model.eval()
        model.set_train(False)
        for _, (data, label) in enumerate(data_loader):
            data = data.numpy()
            label = label.numpy()
            data = mindspore.Tensor(data)
            label = mindspore.Tensor(label)
            label_int32 = label.astype(mindspore.int32)
            
            logits = model(data)
            loss = criterion(logits, label_int32)
            loss = loss.asnumpy()
            total_loss += loss.item() * logits.shape[0]

            # _, pred = torch.max(logits, dim=1)  # replace softmax with max function, same impacts
            max = ops.ArgMaxWithValue(axis=1)
            _, pred = max(logits) # replace softmax with max function, same impacts
#             pred = pred.cpu().detach().numpy().reshape((-1, 1))
#             label = label.cpu().detach().numpy().reshape((-1, 1))
            pred = pred.asnumpy().reshape((-1, 1))
            label = label.asnumpy().reshape((-1, 1))
            predict_label.append(pred)
            true_label.append(label)
        predict_label = np.concatenate(predict_label, axis=0).reshape(-1).astype(np.int64)
        true_label = np.concatenate(true_label, axis=0).reshape(-1).astype(np.int64)
        eval_loss = total_loss / predict_label.shape[0]

        f1 = semeval_scorer(predict_label, true_label)
        return f1, eval_loss, predict_label
