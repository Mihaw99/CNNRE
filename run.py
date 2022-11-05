#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.7.5

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

import numpy as np
from config import Config
from utils import WordEmbeddingLoader, RelationLoader, SemEvalDataLoader
from model import CNN
from evaluate import Eval


def print_result(predict_label, id2rel, start_idx=8001):
    with open('predicted_result.txt', 'w', encoding='utf-8') as fw:
        for i in range(0, predict_label.shape[0]):
            fw.write('{}\t{}\n'.format(
                start_idx+i, id2rel[int(predict_label[i])]))


def train(model, criterion, loader, config, loader0, optimizer):
#     train_loader, dev_loader, _ = loader
#     train_loader0, dev_loader0, _ = loader0
    # optimizer = optim.Adam(model.parameters(), lr=config.lr,
    #                        weight_decay=config.L2_decay)
    # print(model)
    train_loader, dev_loader, test_loader = loader
    train_loader0, dev_loader0, test_loader0 = loader0
    

    eval_tool = Eval(config)
    max_f1 = -float('inf')

    def forward_fn(data, label):
        logits = model(data)
        loss = criterion(logits, label)
        return loss, logits
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss,_), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss


    for epoch in range(1, config.epoch+1):
        model.set_train()
        for step, (data, label) in enumerate(train_loader0):
            data = data.numpy()
            label = label.numpy()
            data = mindspore.Tensor(data)
            label = mindspore.Tensor(label)
            label_int32 = label.astype(mindspore.int32)

            loss = train_step(data, label_int32)

            logits = model(data)
            loss = criterion(logits, label_int32)

        _, train_loss, _ = eval_tool.evaluate(model, criterion, train_loader0)
        f1, dev_loss, _ = eval_tool.evaluate(model, criterion, dev_loader0)

        print('[%03d] train_loss: %.3f | dev_loss: %.3f | micro f1 on dev: %.4f'
              % (epoch, train_loss, dev_loss, f1), end=' ')
        if f1 > max_f1:
            max_f1 = f1
            # torch.save(model.state_dict(), os.path.join(
            #     config.model_dir, 'model.pkl'))
            mindspore.save_checkpoint(model, "model.ckpt")
            print('>>> save models!')
        else:
            print()    

#     size = train_loader.get_dataset_size()
#     # print("size: ", size)
# #     model.set_train()
#     for batch, (data, label) in enumerate(train_loader0):
#         model.set_train()
#         data = data.numpy()
#         label = label.numpy()
#         data = mindspore.Tensor(data)
#         label = mindspore.Tensor(label)
#         label_int32 = label.astype(mindspore.int32)

#         loss = train_step(data, label_int32, test_loss)
            
#         if batch % 10 == 0:
#             loss, current = loss.asnumpy(), batch
#             print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
#             # return min_loss    



def test(model, criterion, loader, config, loader0):
    _, _, test_loader = loader
    _, _, test_loader0 = loader0

    num_batches = test_loader.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for batch, (data, label) in enumerate(test_loader0):
        data = data.numpy()
        label = label.numpy()
        data = mindspore.Tensor(data)
        label = mindspore.Tensor(label)
        label_int32 = label.astype(mindspore.int32)

        pred = model(data)
        total += len(data)
        test_loss += criterion(pred, label_int32).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # return 100*correct

def test_old(model, criterion, loader, config, loader0):
    print('--------------------------------------')
    print('start test ...')

    _, _, test_loader0 = loader0
    # model.load_state_dict(torch.load(
    #     os.path.join(config.model_dir, 'model.pkl')))
    param_dict=mindspore.load_checkpoint("model.ckpt")
    param_not_load = mindspore.load_param_into_net(model, param_dict)

    eval_tool = Eval(config)
    f1, test_loss, predict_label = eval_tool.evaluate(
        model, criterion, test_loader0)
    print('test_loss: %.3f | micro f1 on test:  %.4f' % (test_loss, f1))
    return predict_label


if __name__ == '__main__':
    config = Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    loader = SemEvalDataLoader(rel2id, word2id, config)

    train_loader, dev_loader = None, None
    if config.mode == 1:  # train mode
        train_loader,train_loader0 = loader.get_train()
        dev_loader,dev_loader0 = loader.get_dev()
    test_loader,test_loader0 = loader.get_test()
    loader = [train_loader, dev_loader, test_loader]
    loader0 = [train_loader0, dev_loader0, test_loader0]
    print('finish!')

    print('--------------------------------------')
    model = CNN(word_vec=word_vec, class_num=class_num, config=config)
    # for n,p in model.parameters_and_names():
    #     if p.ndim > 1:
    #         if(n == "pos1_embedding.embedding_table" or "pos2_embedding.embedding_table"
    #                 or "conv.weight" or ):
    #             p = initializer(XavierNormal(), p.shape)
            
    #         p = initialzer(XavierUniform(), p.shape, dtype=mindspore.float32)
    # criterion = nn.CrossEntropyLoss()
    # print("model.parameters_and_names: ", model.parameters_and_names)
    # criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config.lr,
                        weight_decay=config.L2_decay)

    # print(model)
    print('traning model parameters:')
    for name, param in model.parameters_and_names():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    print('--------------------------------------')
    print('start to train the model ...')

    # min_loss = float('inf')
    # max_ac = -float('inf')
#     for epoch in range(1, config.epoch+1):
#         print(f"Epoch {epoch}\n-------------------------------")
#         train(model, criterion, loader, config, loader0, optimizer)
#         test(model, criterion, loader, config, loader0)

    if config.mode == 1:  # train mode
        train(model, criterion, loader, config, loader0, optimizer)
    predict_label = test_old(model, criterion, loader, config, loader0)
    print_result(predict_label, id2rel)
