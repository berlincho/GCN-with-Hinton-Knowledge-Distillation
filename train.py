from __future__ import division
from __future__ import print_function

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy, load_data_new, preprocess_features, preprocess_adj
from models import GCN
from params import args
from kd_losses import *

# Model and optimizer
class Train:
    def __init__(self, args):
        self.args = args
        self.best_teacher_val, self.best_student_val = 0, 0
        self.teacher_state, self.student_state = None, None
        self.load_data()
        # Model Initialization
        self.t_model = GCN(nfeat=self.features.shape[1],
                            nhid=self.args.hidden,
                            nclass=self.labels.max().item() + 1,
                            dropout=self.args.dropout)  
        self.s_model = GCN(nfeat=self.features.shape[1],
                            nhid=self.args.hidden,
                            nclass=self.labels.max().item() + 1,
                            dropout=self.args.dropout)
        # Setup loss criterion
        self.criterionTeacher = nn.CrossEntropyLoss()
        self.criterionStudent = nn.CrossEntropyLoss()
        self.criterionStudentKD = SoftTarget(args.T)
        # Setup Training Optimizer
        self.optimizerTeacher = optim.Adam(self.t_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.optimizerStudent = optim.Adam(self.s_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # CUDA setup
        if self.args.cuda:
            self.t_model.cuda()
            self.s_model.cuda()
            self.features = self.features.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()
            
    def load_data(self):
        # Load data
        self.adj, self.features, self.labels, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask, self.idx_train, self.idx_val, self.idx_test = load_data_new(self.args.data)
        print('Load ', self.args.data)
        print('adj:', self.adj.shape)
        print('features:', self.features.shape)
        print('y:', self.y_train.shape, self.y_val.shape, self.y_test.shape)
        print('mask:', self.train_mask.shape, self.val_mask.shape, self.test_mask.shape)

        # D^-1@X
        self.features = preprocess_features(self.features) # [49216, 2], [49216], [2708, 1433]
        self.supports = preprocess_adj(self.adj)

        device = torch.device('cuda')
        i = torch.from_numpy(self.features[0]).long().to(device)
        v = torch.from_numpy(self.features[1]).to(device)
        self.features = torch.sparse.FloatTensor(i.t(), v, self.features[2]).to(device)

        i = torch.from_numpy(self.supports[0]).long().to(device)
        v = torch.from_numpy(self.supports[1]).to(device)
        self.adj = torch.sparse.FloatTensor(i.t(), v, self.supports[2]).float().to(device)

        print('features:', self.features)
        print('adj:', self.adj)
        
    def pre_train_teacher(self,epoch):
        t = time.time()
        self.t_model.train()
        self.optimizerTeacher.zero_grad()
        output = self.t_model(self.features, self.adj)
        loss_train = self.criterionTeacher(output[self.idx_train], self.labels[self.idx_train])
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizerTeacher.step()

        if not self.args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.t_model.eval()
            output = self.t_model(self.features, self.adj)

        loss_val = self.criterionTeacher(output[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        # Do validation
        if acc_val > self.best_teacher_val:
            self.best_teacher_val = acc_val
            self.teacher_state = {
                'state_dict': self.t_model.state_dict(),
                'best_val': acc_val,
                'best_epoch': epoch+1,
                'optimizer': self.optimizerTeacher.state_dict(),
            }
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
        
    def train_student(self, epoch):
        t = time.time()
        self.s_model.train()
        self.optimizerStudent.zero_grad()
        output = self.s_model(self.features, self.adj)
        soft_target = self.t_model(self.features, self.adj)
        loss_train = self.criterionStudent(output[self.idx_train], self.labels[self.idx_train]) * (1-self.args.lambda_kd) + self.criterionStudentKD(output, soft_target) * self.args.lambda_kd
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizerStudent.step()
        # Do validation
        loss_val = self.criterionStudent(output[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        if acc_val > self.best_student_val:
            self.best_student_val = acc_val
            self.student_state = {
                'state_dict': self.s_model.state_dict(),
                'best_val': acc_val,
                'best_epoch': epoch+1,
                'optimizer': self.optimizerStudent.state_dict(),
            }
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
        
    def test(self, ts='teacher'):
        if ts == 'teacher':
            model = self.t_model 
            criterion = self.criterionTeacher
        elif ts == 'student':
            model = self.s_model
            criterion = self.criterionStudent
        model.eval()
        output = model(self.features, self.adj)
        loss_test = criterion(output[self.idx_test], self.labels[self.idx_test])
        acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        print("{ts} Test set results:".format(ts=ts),
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    def save_checkpoint(self, filename='./.checkpoints/'+args.name, ts='teacher'):
        print('Save {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'teacher':  
            torch.save(self.teacher_state, filename)
            print('Successfully saved teacher model\n...')
        elif ts == 'student':
            torch.save(self.student_state, filename)
            print('Successfully saved student model\n...')
        
        
    def load_checkpoint(self, filename='./.checkpoints/'+args.name, ts='teacher'):
        print('Load {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'teacher':
            load_state = torch.load(filename)
            self.t_model.load_state_dict(load_state['state_dict'])
            self.optimizerTeacher.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded teacher model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'].item())
        elif ts == 'student':
            load_state = torch.load(filename)
            self.s_model.load_state_dict(load_state['state_dict'])
            self.optimizerStudent.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded student model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'].item())      