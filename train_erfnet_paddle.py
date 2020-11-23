import os
import time
import cv2
import utils.transforms as tf
import numpy as np
import models
import datasets as ds
from options.options import parser
import paddle.fluid as fluid
from paddle.fluid.io import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(args.save_dir)
best_mIoU = 0
start_epoch = 0


def main():
    global best_mIoU, start_epoch

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.dataset == 'LaneDet':
        num_class = 20
        ignore_label = 255
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    # get places
    places = fluid.cuda_places()

    with fluid.dygraph.guard():
        model = models.ERFNet(num_class, [args.img_height, args.img_width])
        input_mean = model.input_mean
        input_std = model.input_std

        # Data loading code
        train_dataset = ds.LaneDataSet(
            dataset_path='datasets/PreliminaryData',
            data_list=args.train_list,
            transform=[
                tf.GroupRandomScale(size=(int(args.img_width), int(args.img_width * 1.2)),
                                    interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                tf.GroupRandomCropRatio(size=(args.img_width, args.img_height)),
                tf.GroupNormalize(mean=(input_mean, (0,)), std=(input_std, (1,))),
            ]
        )

        train_loader = DataLoader(
            train_dataset,
            places=places[0],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True
        )

        val_dataset = ds.LaneDataSet(
            dataset_path='datasets/PreliminaryData',
            data_list=args.train_list,
            transform=[
                tf.GroupRandomScale(size=args.img_width, interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                tf.GroupNormalize(mean=(input_mean, (0,)), std=(input_std, (1,))),
            ],
            is_val=False
        )

        val_loader = DataLoader(
            val_dataset,
            places=places[0],
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
        )

        # define loss function (criterion) optimizer and evaluator
        weights = [1.0 for _ in range(num_class)]
        weights[0] = 0.25
        weights = fluid.dygraph.to_variable(np.array(weights, dtype=np.float32))
        criterion = fluid.dygraph.NLLLoss(weight=weights, ignore_index=ignore_label)
        evaluator = EvalSegmentation(num_class, ignore_label)

        optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=fluid.dygraph.CosineDecay(
                                                                    args.lr, len(train_loader), args.epochs),
                                                      momentum=args.momentum,
                                                      parameter_list=model.parameters(),
                                                      regularization=fluid.regularizer.L2Decay(
                                                          regularization_coeff=args.weight_decay))

        if args.resume:
            print(("=> loading checkpoint '{}'".format(args.resume)))
            start_epoch = int(''.join([x for x in args.resume.split('/')[-1] if x.isdigit()]))
            checkpoint, optim_checkpoint = fluid.load_dygraph(args.resume)
            model.load_dict(checkpoint)
            optimizer.set_dict(optim_checkpoint)
            print(("=> loaded checkpoint (epoch {})".format(start_epoch)))
        else:
            try:
                checkpoint, _ = fluid.load_dygraph(args.weight)
                model.load_dict(checkpoint)
                print("=> pretrained model loaded successfully")
            except:
                print(("=> no pretrained model found at '{}'".format(args.weight)))

        for epoch in range(start_epoch, args.epochs):
            # train for one epoch
            loss = train(train_loader, model, criterion, optimizer, epoch)

            # writer.add_scalar('lr', optimizer.current_step_lr(), epoch + 1)

            if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
                save_checkpoint(model.state_dict(), epoch)
                save_checkpoint(optimizer.state_dict(), epoch)

            # evaluate on validation set
            if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
                mIoU = validate(val_loader, model, evaluator, epoch)

                # remember best mIoU
                is_best = mIoU > best_mIoU
                best_mIoU = max(mIoU, best_mIoU)
                if is_best:
                    tag_best(epoch, best_mIoU)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    epoch_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)  # output_mid
        output = fluid.layers.log_softmax(output, axis=1)
        loss = criterion(output, target)

        # measure accuracy and record loss
        epoch_losses.update(loss.numpy()[0], input.shape[0])
        losses.update(loss.numpy()[0], input.shape[0])
        # writer.add_scalar('Loss/train', loss.numpy()[0], i + epoch * len(train_loader))

        # compute gradient and do SGD step
        model.clear_gradients()
        loss.backward()
        optimizer.minimize(loss)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], lr: {lr:.5f} Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.4f} ({data_time.avg:.4f}) Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                lr=optimizer.current_step_lr()))
            batch_time.reset()
            data_time.reset()
            losses.reset()

    # writer.add_scalar('Epoch Loss/train', epoch_losses.avg, epoch + 1)
    return epoch_losses.avg


def validate(val_loader, model, evaluator, epoch):
    batch_time = AverageMeter()
    IoU = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = fluid.dygraph.to_variable(input.numpy())
        target = target.numpy()

        # compute output
        output = model(input).numpy()

        # measure accuracy and record loss
        pred = np.argmax(output, 1)
        IoU.update(evaluator(pred, target))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % (args.print_freq * 10) == 0:
            acc = np.sum(np.diag(IoU.sum)) / float(np.sum(IoU.sum))
            mIoU = np.mean(np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum)))
            print('Test: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Pixels Acc {acc:.3f} mIoU {mIoU:.3f}'.format(
                i, len(val_loader), batch_time=batch_time, acc=acc, mIoU=mIoU))

    acc = np.sum(np.diag(IoU.sum)) / float(np.sum(IoU.sum))
    mIoU = np.mean(np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum)))
    # writer.add_scalar('acc', acc, epoch + 1)
    # writer.add_scalar('mIoU', mIoU, epoch + 1)
    print(('Testing Results: Pixels Acc {acc:.3f}\tmIoU {mIoU:.3f} ({bestmIoU:.4f})'.format(
        acc=acc, mIoU=mIoU, bestmIoU=max(mIoU, best_mIoU))))
    del batch_time
    del IoU
    return mIoU


def save_checkpoint(state, epoch):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    filename = os.path.join(args.save_dir, '_'.join([args.method.lower(), 'ep{}'.format(epoch + 1)]))
    fluid.dygraph.save_dygraph(state, filename)


def tag_best(epoch, best_mIoU):
    info = '_'.join([args.method.lower(), 'ep{}.pth'.format(epoch + 1)])
    info += ' best_mIoU: {}'.format(best_mIoU)
    save_file = os.path.join(args.save_dir, "best_model")
    with open(save_file, "w") as f:
        f.write(info)  # pyre-ignore


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert pred.shape == gt.shape
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = gt != self.ignore_label
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class ** 2).reshape(self.num_class, self.num_class)
        return hs


if __name__ == '__main__':
    args = parser.parse_args()
    main()
