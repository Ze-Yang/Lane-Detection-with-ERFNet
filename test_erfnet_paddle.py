import os
import time
import cv2
import numpy as np
import models
import datasets as ds
from datasets.lane_det import collate_fn
from options.options import parser
import paddle.fluid as fluid
from paddle.fluid.io import DataLoader

best_mIoU = 0


def main():
    global args, best_mIoU
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.dataset == 'LaneDet':
        num_class = 20
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    # get places
    places = fluid.cuda_places()

    with fluid.dygraph.guard():
        model = models.ERFNet(num_class, [576, 1024])
        input_mean = model.input_mean
        input_std = model.input_std

        if args.resume:
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint, _ = fluid.load_dygraph(args.resume)
            model.load_dict(checkpoint)
            print("=> checkpoint loaded successfully")
        else:
            print(("=> loading checkpoint '{}'".format('trained/ERFNet_trained')))
            checkpoint, _ = fluid.load_dygraph('trained/ERFNet_trained')
            model.load_dict(checkpoint)
            print("=> default checkpoint loaded successfully")

        # Data loading code
        test_dataset = ds.LaneDataSet(
            dataset_path='datasets/PreliminaryData',
            data_list=args.val_list,
            transform=[
                lambda x: cv2.resize(x, (1024, 576)),
                lambda x: x - np.asarray(input_mean)[None, None, :] / np.array(input_std)[None, None, :],
            ]
        )

        test_loader = DataLoader(
            test_dataset,
            places=places[0],
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_fn
        )

        ### evaluate ###
        mIoU = validate(test_loader, model)
        # print('mIoU: {}'.format(mIoU))
    return


def validate(test_loader, model):

    batch_time = AverageMeter()
    # IoU = AverageMeter()
    mIoU = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, img_name, img_size) in enumerate(test_loader):

        # compute output
        output = model(input).numpy()

        # measure accuracy and record loss
        pred = np.argmax(output, 1)

        img_name = img_name.numpy().tolist()
        img_size = img_size.numpy().tolist()

        for j, (name, size) in enumerate(zip(img_name, img_size)):
            # scale = size[1] / pred.shape[2]
            # img = cv2.resize(pred[j], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            # img = np.pad(img, ((size[0] - img.shape[0], 0), (0, 0)))
            img = cv2.resize(pred[j], tuple(size[:2][::-1]), interpolation=cv2.INTER_NEAREST)
            img = img.astype(np.uint8)
            file_name = os.path.join(args.output_dir, str(name) + '.png')
            cv2.imwrite(file_name, img)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                i + 1, len(test_loader), batch_time=batch_time)))

    print('finished, #test:{}'.format(i + 1))

    return mIoU


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


if __name__ == '__main__':
    main()
