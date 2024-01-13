import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
import numpy as np
import models
from models import sync_bn
import dataset as ds
from options.options import parser
import numpy as np
import collections

best_mIoU = 0


def main():
    global args, best_mIoU
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ' '

    if args.dataset == 'VOCAug' or args.dataset == 'VOC2012' or args.dataset == 'COCO':
        num_class = 21
        ignore_label = 255
        scale_series = [10, 20, 30, 60]
    elif args.dataset == 'Cityscapes':
        num_class = 19
        ignore_label = 255 # 0
        scale_series = [15, 30, 45, 90]
    elif args.dataset == 'ApolloScape':
        num_class = 37 # merge the noise and ignore labels
        ignore_label = 255 # 0
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = models.PSPNet(num_class, base_model=args.arch, dropout=args.dropout, partial_bn=not args.no_partialbn)
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict_cpu = collections.OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                state_dict_cpu[k[7:]] = v
            args.start_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            torch.nn.Module.load_state_dict(model, state_dict_cpu)
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code

    test_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("ApolloScape", "VOCSingle") + 'DataSet')(data_list=args.val_list, transform=[
            torchvision.transforms.Compose([
            # tf.GroupRandomScaleRatio(size=(1692, 1692, 505, 505), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),]), 
            ]), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # define loss function (criterion) optimizer and evaluator
    weights = [1.0 for _ in range(37)]
    weights[0] = 0.05
    weights[36] = 0.05
    class_weights = torch.FloatTensor(weights)
    criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights)
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    evaluator = EvalSegmentation(num_class, ignore_label)

    ### evaluate ###
    validate(test_loader, model, criterion, 0, evaluator)
    return

def cal_model_output(model, input_img):

    input_var = torch.autograd.Variable(input_img, volatile=True)        
    output = model(input_var)
    pred = output.data.cpu().numpy()

    return pred


def validate(val_loader, model, criterion, iter, evaluator, logger=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()
    mIoU = 0
    val_img_list = []
    model.eval()
    end = time.time()
    
    for i, (input_, img_name) in enumerate(val_loader): #, input_5
        pred = cal_model_output(model, input_)
        pred = pred.transpose(0, 2, 3, 1)
        pred = np.argmax(pred, axis=3).astype(np.uint8)
        pred = pred + 1
        for cnt in range(len(img_name)):
            np.save('road05_tmp/' + img_name[cnt].split('/')[6].replace('jpg', 'npy'), pred[cnt]) #split('/')[5]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time)))

    print('finished, #test:{}'.format(i))
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


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs


if __name__ == '__main__':
    main()
