import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import datetime
import numpy as np

""" Parts of the U-Net model """
class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, 
                 dilation=2, nlayers=2, batchnorm=True, 
                 batchnormfirst=True, padding=1):
        super().__init__()
        layers = []
        for i in range(nlayers):
            channels = in_channels if i == 0 else out_channels
            layers.append(nn.Conv1d(channels, out_channels, kernel_size=kernel_size, 
                        padding=int(dilation*(kernel_size-1)/2), dilation=dilation))
            if batchnorm and batchnormfirst:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if batchnorm and not batchnormfirst:
                layers.append(nn.BatchNorm1d(out_channels))
        self.multi_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.multi_conv(x)


class Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 batchnorm=True, batchnormfirst=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding=int((kernel_size-1)/2), dilation=1))
        if batchnorm and batchnormfirst:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if batchnorm and not batchnormfirst:
            layers.append(nn.BatchNorm1d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, outconv_kernel=3,
                 nlayers=2, batchnorm=True, batchnormfirst=False):
        super(OutConv, self).__init__()
        self.conv_softmax = nn.Sequential(MultiConv(in_channels, out_channels, 
                                                     kernel_size=outconv_kernel, dilation=1,
                                                     padding=int((outconv_kernel-1)/2), nlayers=nlayers, batchnorm=batchnorm, 
                                                     batchnormfirst=batchnormfirst),
                                          nn.Conv1d(in_channels, n_classes, kernel_size=1, padding=0),
                                          nn.Softmax(dim=1))
    def forward(self, x):
        return self.conv_softmax(x)


""" Full assembly of the parts to form the complete network """
class hypoptUNet(nn.Module):
    def __init__(self, n_features: int = 4, init_channels: int = 16, 
                 n_classes: int = 4, depth: int = 4, enc_kernel: int = 5,
                 dec_kernel: int = 5, outconv_kernel: int = 5, 
                 dil_rate = 2, pools: list = [2, 2, 2, 2], 
                 pooling: str = 'max', enc_conv_nlayers: int = 2,
                 dec_conv_nlayers: int = 2, bottom_conv_nlayers: int = 2,
                 out_nlayers: int = 2, X_padtoken: int = 0, 
                 y_padtoken: int = 10, batchnorm: bool = True, 
                 batchnormfirst: bool = False, channel_multiplier: int = 2,
                 device: str = 'cpu'):
        super(hypoptUNet, self).__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.depth = depth
        self.pools = pools
        self.decoder_scale_factors = pools[::-1]
        self.enc_kernel = enc_kernel
        self.dec_kernel = dec_kernel
        self.outconv_kernel = outconv_kernel
        self.dil_rate = dil_rate
        self.enc_conv_nlayers = enc_conv_nlayers
        self.dec_conv_nlayers = dec_conv_nlayers
        self.bottom_conv_nlayers = bottom_conv_nlayers
        self.out_nlayers = out_nlayers
        self.batchnorm = batchnorm
        self.batchnormfirst = batchnormfirst
        self.channel_multiplier = channel_multiplier
        self.pooling = nn.MaxPool1d if pooling=='max' else nn.AvgPool1d
        self.X_padtoken = X_padtoken
        self.y_padtoken = y_padtoken
        self.device = device

        self.module_list = nn.ModuleList()
        in_channels = n_features
        out_channels = init_channels
        res_channels = []
        for i in range(depth):
            self.module_list.append(MultiConv(int(in_channels), 
                                            int(out_channels), 
                                            kernel_size=self.enc_kernel, 
                                            dilation=self.dil_rate,
                                            nlayers=enc_conv_nlayers, 
                                            batchnorm=batchnorm, 
                                            batchnormfirst=batchnormfirst))
            in_channels = out_channels
            res_channels.append(out_channels)
            out_channels *= channel_multiplier
        self.module_list.append(MultiConv(int(in_channels), 
                                          int(out_channels), 
                                          kernel_size=self.enc_kernel, 
                                          dilation=self.dil_rate,
                                          nlayers=bottom_conv_nlayers, 
                                          batchnorm=batchnorm, 
                                          batchnormfirst=batchnormfirst))
        in_channels = out_channels
        out_channels /= channel_multiplier
        for i in range(depth):
            self.module_list.append(Conv_layer(int(in_channels), 
                                                int(out_channels), 
                                                kernel_size=self.dec_kernel,
                                                batchnorm=batchnorm, 
                                                batchnormfirst=batchnormfirst))
    
            merge_channels = out_channels + res_channels[::-1][i]

            self.module_list.append(MultiConv(int(merge_channels), 
                                              int(out_channels), 
                                              kernel_size=self.dec_kernel, 
                                              dilation=1,
                                              nlayers=dec_conv_nlayers, 
                                              batchnorm=batchnorm, 
                                              batchnormfirst=batchnormfirst))
            in_channels = out_channels
            if i != self.depth-1:
                out_channels /= channel_multiplier

        self.module_list.append(OutConv(int(out_channels), 
                                int(out_channels), 
                                self.n_classes, 
                                outconv_kernel=self.outconv_kernel,
                                nlayers=out_nlayers, 
                                batchnorm=batchnorm, 
                                batchnormfirst=batchnormfirst))
        self.decoder_list = self.module_list[(depth+1):(depth+2*depth+1)]

        self.to(device)

    def concat(self, x1, x2):
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffX, 0])
        x = torch.cat([x2, x1], dim=1)
        return x

    def match_x1_to_x2(self, x1=None, x2=None, value=0):
        diffX = x2.size()[2] - x1.size()[2]
        x = F.pad(x1, [diffX, 0], value=value)
        return x

    def predict(self, test_loader, temperature=1):
        with torch.no_grad():
            masked_ys = []
            masked_preds = []
            masked_argmaxs = []
            device = next(self.parameters()).device
            for xb in tqdm(test_loader):
                x, y = xb[0].to(device), xb[1].to(device)
                out = self.forward((x,y), inference=True, temperature=temperature)
                _, _, _, masked_y, masked_pred, masked_argmax  = out
                masked_ys.append(masked_y)
                masked_preds.append(masked_pred)
                masked_argmaxs.append(masked_argmax)
        masked_argmaxs = [i for s in masked_argmaxs for i in s]
        masked_preds = [i for s in masked_preds for i in s]
        masked_ys = [i for s in masked_ys for i in s]
        return masked_argmaxs, masked_preds, masked_ys

    def simple_predict(self, xb, temperature=1):
        with torch.no_grad():
            x, y = xb
            out = self.forward((x,y), inference=True, temperature=temperature)
            _, _, _, masked_y, masked_pred, masked_argmax  = out
        return masked_argmax, masked_pred, masked_y

    def forward(self, xb, inference=False, temperature=1):
        x, y = xb
        input = xb[0]
        residuals_list = []
        for i in range(self.depth):
            pool = self.pools[i]
            res = self.module_list[i](x)
            x = self.pooling(pool)(res)
            residuals_list.append(res)
        x = self.module_list[self.depth](x)
        residual = residuals_list[::-1]
        for i in range(0, self.depth*2, 2):
            scale_factor = self.decoder_scale_factors[i//2]
            up = nn.Upsample(scale_factor=scale_factor, mode='nearest')(x)
            x = self.decoder_list[i](up)
            merged = self.concat(x, residual[i//2])
            x = self.decoder_list[i+1](merged)
        merged = self.match_x1_to_x2(x1=x, x2=input, value=0)
        pred = self.module_list[-1](merged)
        loss = 0
        acc = 0
        criterion = nn.CrossEntropyLoss()
        masked_ys = []
        masked_preds = []
        masked_argmax = []
        for i in range(len(y)):
            mask_idx = sum(y[i].ge(self.y_padtoken))
            masked_y = y[i][mask_idx:].unsqueeze(0)
            masked_pred = pred[i][:,mask_idx:].unsqueeze(0)
            loss += criterion(masked_pred, masked_y.long())
            acc += torch.sum(masked_pred.argmax(1) == masked_y, 1)/masked_y.shape[1]            
            masked_ys.append(masked_y.cpu().squeeze(0).numpy())
            masked_preds.append(masked_pred.detach().cpu().squeeze(0).numpy())
            masked_argmax.append(masked_pred.argmax(1).squeeze(0).detach().cpu().numpy())
        loss /= y.shape[0]
        acc /= y.shape[0]
        if inference:
            return loss, acc, pred, masked_ys, masked_preds, masked_argmax
        else:
            return loss, acc, pred


class encoderhypoptUNet(nn.Module):
    def __init__(self, n_features: int = 4, init_channels: int = 16, 
                 n_classes: int = 4, depth: int = 4, enc_kernel: int = 5,
                 dil_rate = 2, pools: list = [2, 2, 2, 2], pooling: str = 'max', 
                 enc_conv_nlayers: int = 2, bottom_conv_nlayers: int = 2,
                 X_padtoken: int = 0, y_padtoken: int = 10, batchnorm: bool = True, 
                 batchnormfirst: bool = False, channel_multiplier: int = 2,
                 device: str = 'cpu'):
        super(hypoptUNet, self).__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.depth = depth
        self.pools = pools
        self.enc_kernel = enc_kernel
        self.dil_rate = dil_rate
        self.enc_conv_nlayers = enc_conv_nlayers
        self.bottom_conv_nlayers = bottom_conv_nlayers
        self.batchnorm = batchnorm
        self.batchnormfirst = batchnormfirst
        self.channel_multiplier = channel_multiplier
        self.pooling = nn.MaxPool1d if pooling=='max' else nn.AvgPool1d
        self.X_padtoken = X_padtoken
        self.y_padtoken = y_padtoken
        self.device = device

        self.module_list = nn.ModuleList()
        in_channels = n_features
        out_channels = init_channels
        res_channels = []
        for i in range(depth):
            self.module_list.append(MultiConv(int(in_channels), 
                                            int(out_channels), 
                                            kernel_size=self.enc_kernel, 
                                            dilation=self.dil_rate,
                                            nlayers=enc_conv_nlayers, 
                                            batchnorm=batchnorm, 
                                            batchnormfirst=batchnormfirst))
            in_channels = out_channels
            res_channels.append(out_channels)
            out_channels *= channel_multiplier
        self.module_list.append(MultiConv(int(in_channels), 
                                          int(out_channels), 
                                          kernel_size=self.enc_kernel, 
                                          dilation=self.dil_rate,
                                          nlayers=bottom_conv_nlayers, 
                                          batchnorm=batchnorm, 
                                          batchnormfirst=batchnormfirst))
        
        self.to(device)

    def concat(self, x1, x2):
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffX, 0])
        x = torch.cat([x2, x1], dim=1)
        return x

    def match_x1_to_x2(self, x1=None, x2=None, value=0):
        diffX = x2.size()[2] - x1.size()[2]
        x = F.pad(x1, [diffX, 0], value=value)
        return x

    def forward(self, xb):
        x, _ = xb
        residuals_list = []
        for i in range(self.depth):
            pool = self.pools[i]
            res = self.module_list[i](x)
            x = self.pooling(pool)(res)
            residuals_list.append(res)
        x = self.module_list[self.depth](x)
        
        return x