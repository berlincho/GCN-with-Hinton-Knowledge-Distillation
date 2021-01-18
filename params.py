import time
import torch
import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="testrun",
                    help='Provide a test name.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--momentum', type=float, default=0.9, 
                    help='momentum')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--T', type=float, default=4.0,
                    help='temperature for ST')
parser.add_argument('--lambda_kd', type=float, default=0.7,
                    help='trade-off parameter for kd loss')

parser.add_argument('--data', type=str, default="cora", help='dataset.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')