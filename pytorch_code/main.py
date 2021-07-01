#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample',
                    help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int,
                    default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int,
                    default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30,
                    help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1,
                    help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3,
                    help='the number of steps after which the learning rate decay')
# [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--step', type=int, default=1,
                    help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10,
                    help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true',
                    help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--topk', type=int, default=20,
                    help='number of top score items selected for calculating recall and mrr metrics')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(
        open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.test:
        test_data = pickle.load(
            open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    else:
        train_data, test_data = split_validation(train_data, opt.valid_portion)

    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node))

    if opt.test:
        start = time.time()
        ckpt = torch.load('../datasets/' + opt.dataset +
                          '/' + 'latest_checkpoint.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        recall, mrr = validate(model, test_data)

        result = "Test: Recall@{}: {:.4f}, MRR@{}: {:.4f}".format(
            opt.topk, recall, opt.topk, mrr)
        print(result)
        end = time.time()

        runtime = "Run time: %f s" % (end - start)

        file_path = "../logs/" + opt.dataset + "/test" + ".txt"
        with open(file_path, "a") as f:
            f.write(str(result))
            f.write('\n')
            f.write(str(runtime))

        print(runtime)

        return

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):

        start1 = time.time()
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        train(model, train_data, optimizer, epoch, opt.epoch,
              loss_function, scheduler, log_aggr=200)
        hit, mrr = validate(model, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        result = "Validation: Recall@{}: {:.4f} at Epoch: {}, MRR@{}: {:.4f} at Epoch: {}".format(opt.topk,
                                                                                                  best_result[0], best_epoch[0], opt.topk, best_result[1], best_epoch[1])

        print(result)

        end1 = time.time()

        runtime = "Run time: %f s" % (end1 - start1)

        file_path = "../logs/" + opt.dataset + "/" + str(opt.topk) + ".txt"
        with open(file_path, "a") as f:
            f.write(str(result))
            f.write('\n')
            f.write(str(runtime))
            f.write('\n')

        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, '../datasets/' + opt.dataset +
                   '/' + 'latest_checkpoint.pth.tar')

        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


def train(model, train_data, optimizer, epoch, num_epochs, loss_function, scheduler, log_aggr=1):
    scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = loss_function(scores, targets - 1)
        loss.backward()
        optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)


def validate(model, test_data):
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    with torch.no_grad():
        for i in slices:
            targets, scores = forward(model, i, test_data)
            sub_scores = scores.topk(opt.topk)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                hit.append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr


if __name__ == '__main__':
    main()
