import os
import torch
import torch.nn as nn

from .attention_augmented_conv import AugmentedConv

def getInput(args, data):
    input_list = [data['img']]
    if args.in_light: input_list.append(data['dirs'])
    if args.in_mask:  input_list.append(data['m'])
    return input_list

def parseData(args, sample, timer=None, split='train'):
    img, normal, mask = sample['img'], sample['normal'], sample['mask']
    ints = sample['ints']
    if args.in_light:
        dirs = sample['dirs'].expand_as(img)
    else: # predict lighting, prepare ground truth
        n, c, h, w = sample['dirs'].shape
        dirs_split = torch.split(sample['dirs'].view(n, c), 3, 1)
        dirs = torch.cat(dirs_split, 0)
    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        img, normal, mask = img.cuda(), normal.cuda(), mask.cuda()
        dirs, ints = dirs.cuda(), ints.cuda()
        if timer: timer.updateTime('ToGPU')
    data = {'img': img, 'n': normal, 'm': mask, 'dirs': dirs, 'ints': ints}
    return data 

def getInputChanel(args):
    args.log.printWrite('[Network Input] Color image as input')
    c_in = 3
    if args.in_light:
        args.log.printWrite('[Network Input] Adding Light direction as input')
        c_in += 3
    if args.in_mask:
        args.log.printWrite('[Network Input] Adding Mask as input')
        c_in += 1
    args.log.printWrite('[Network Input] Input channel: {}'.format(c_in))
    return c_in

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records} # 'args': args}
    torch.save(state,   os.path.join(save_path, 'checkp_{}.pth.tar'.format(epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_{}_rec.pth.tar'.format(epoch)))

def conv_ReLU(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.ReLU(inplace=True)
                )

def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
                #                   dilation=1, groups=1, bias=True, padding_mode='zeros')
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )

def resConv(batchNorm, in_channels, out_channels, k=3, stride=1, pad=-1):
    #residual function
    residual_function = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels)
    )

    #shortcut
    shortcut = nn.Sequential()

    #the shortcut output dimension is not the same with residual function
    #use 1*1 convolution to match the dimension
    if stride != 1 or in_channels != out_channels:
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    return {"residual" : residual_function, "shortcut" : shortcut}

def augmented_conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                AugmentedConv(in_channels=cin, out_channels=cout, dk=cout//4, dv=cout//4, Nh=4, kernel_size=k, stride=stride),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                AugmentedConv(in_channels=cin, out_channels=cout, dk=cout//4, dv=cout//4, Nh=4, kernel_size=k, stride=stride),
                nn.LeakyReLU(0.1, inplace=True)
                )

def outputConv(cin, cout, k=3, stride=1, pad=1):
    return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True))

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )

def upconv(cin, cout):
    return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )
