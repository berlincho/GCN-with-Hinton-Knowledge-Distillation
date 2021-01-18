import numpy as np
import torch
import time
import os
from train import Train
from params import args

if __name__ == '__main__':
    if not os.path.exists('.checkpoints'):
        os.makedirs('.checkpoints')
    
    # initialization
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train = Train(args)
    
    # pre-train teacher model
    t_total = time.time()
    for epoch in range(args.epochs):
        train.pre_train_teacher(epoch)
    train.save_checkpoint(ts='teacher')

    # load best pre-train teahcer model
    train.load_checkpoint(ts='teacher')
    
    print('\n--------------\n')
    
    # train student model
    t_total = time.time()
    for epoch in range(args.epochs):
        train.train_student(epoch)
    train.save_checkpoint(ts='student')
   
    # test teacher model
    train.test('teacher')

    # test student model
    train.load_checkpoint(ts='student')
    train.test('student')
   