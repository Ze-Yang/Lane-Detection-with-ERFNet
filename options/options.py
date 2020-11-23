import argparse

parser = argparse.ArgumentParser(description="PaddlePaddle implementation of ERFNet for lane detection.")

parser.add_argument('--dataset', type=str, default='LaneDet')
parser.add_argument('--method', type=str, default='ERFNet')
parser.add_argument('--train_list', type=str, default='train_pic')
parser.add_argument('--val_list', type=str, default='testB')

# ========================= Model Configs ==========================
parser.add_argument('--img_height', default=384, type=int, help='height of input images')
parser.add_argument('--img_width', default=1024, type=int, help='width of input images')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay rate (default: 1e-4)')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=10, type=int, metavar='N', help='evaluation frequency (default: 10)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--weight', default='pretrained/ERFNet_pretrained', type=str, metavar='PATH',
                    help='path to initial weights')
parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--save-dir', type=str, default='trained')
parser.add_argument('--save-freq', type=int, default=10)
parser.add_argument('--output-dir', type=str, default='results/result')
