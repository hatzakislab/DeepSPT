import torch
import torch.nn as nn
import torch.nn.functional as F


""" Parts of the U-Net model """
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=2, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Conv_Bn_ReLu(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class Conv_linear(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        )

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super(OutConv, self).__init__()
        self.conv_softmax = nn.Sequential(DoubleConv(in_channels, out_channels, 
                                                     kernel_size=3, dilation=1,
                                                     padding=1),
                                          Conv_linear(in_channels, n_classes, kernel_size=1),
                                          nn.Softmax(dim=1))

    def forward(self, x):
        return self.conv_softmax(x)


""" Full assembly of the parts to form the complete network """
class UNet(nn.Module):
    def __init__(self, n_features: int = 4, init_channels: int = 16, 
                 n_classes: int = 4, depth: int = 4, enc_kernel: int = 5, 
                 dil_rate = 2, pools: list = [10, 8, 6, 4],
                 X_padtoken: int = 0, y_padtoken: int = 10, 
                 device: str = 'cpu'):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.depth = depth
        self.pools = pools
        self.decoder_kernels = pools[::-1]
        self.enc_kernel = enc_kernel
        self.dil_rate = dil_rate
        self.X_padtoken = X_padtoken
        self.y_padtoken = y_padtoken
        self.device = device
        
        self.module_list = nn.ModuleList()
        
        in_channels = n_features
        out_channels = init_channels
        res_channels = []
        for i in range(depth):
            self.module_list.append(DoubleConv(int(in_channels), int(out_channels), kernel_size=self.enc_kernel, dilation=self.dil_rate))
            in_channels = out_channels
            res_channels.append(out_channels)
            out_channels *= 2 
        self.module_list.append(DoubleConv(int(in_channels), int(out_channels), kernel_size=self.enc_kernel, dilation=self.dil_rate))

        in_channels = out_channels
        out_channels /= 2
        for i in range(depth):
            kernel_size = self.decoder_kernels[i]
            self.module_list.append(Conv_Bn_ReLu(int(in_channels), int(out_channels), kernel_size=kernel_size))
            merge_channels = out_channels + res_channels[::-1][i]
            self.module_list.append(DoubleConv(int(merge_channels), int(out_channels), kernel_size=self.enc_kernel, dilation=1))
            in_channels = out_channels
            if i != self.depth-1:
                out_channels /= 2

        self.module_list.append(OutConv(int(out_channels), int(out_channels), self.n_classes))
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

    def raw_predict(self, X, maxlens=0):
        print('Remembered model.eval?')
        if type(X[0])!=torch.tensor:
            X = [torch.tensor(x) for x in X]
        Y = [torch.ones(x.size(0)) for x in X]
        x = torch.stack([nn.ConstantPad1d((maxlens-len(x), 0), self.X_padtoken)(x.T).float() for x in X])
        y = torch.stack([nn.ConstantPad1d((maxlens-len(y), 0), self.y_padtoken)(y.T).float() for y in Y])
        with torch.no_grad():
            _, _, pred = self.forward((x,y))

        masked_pred_list = []
        masked_argmax_list = []
        for i in range(len(y)):
            mask_idx = sum(y[i].ge(self.y_padtoken))
            masked_pred_list.append(pred[i][:,mask_idx:].cpu().numpy())   
            masked_argmax_list.append(pred[i][:,mask_idx:].argmax(0).cpu().numpy())

        return masked_pred_list, masked_argmax_list


    def predict(self, test_loader):
        print('Remembered model.eval?')
        with torch.no_grad():
            masked_ys = []
            masked_preds = []
            masked_argmax = []
            for xb in test_loader:
                x, y = xb
                input = xb[0]
                residuals_list = []
                for i in range(self.depth):
                    pool = self.pools[i]
                    res = self.module_list[i](x)
                    x = nn.MaxPool1d(pool)(res)
                    residuals_list.append(res)
                x = self.module_list[self.depth](x)
                residual = residuals_list[::-1]
                for i in range(0, self.depth*2, 2):
                    up = nn.Upsample(scale_factor=2, mode='nearest')(x)
                    x = self.decoder_list[i](up)
                    merged = self.concat(x, residual[i//2])         
                    x = self.decoder_list[i+1](merged)

                merged = self.match_x1_to_x2(x1=x, x2=input, value=0)
                pred = self.module_list[-1](merged)

                for i in range(len(y)):
                    mask_idx = sum(y[i].ge(self.y_padtoken))
                    masked_ys.append(y[i][mask_idx:].cpu().numpy())
                    masked_preds.append(pred[i][:,mask_idx:].cpu().numpy())   
                    masked_argmax.append(pred[i][:,mask_idx:].argmax(0).cpu().numpy())

        return masked_argmax, masked_preds, masked_ys

    def forward(self, xb):
        x, y = xb
        input = xb[0]
        residuals_list = []
        for i in range(self.depth):
            pool = self.pools[i]
            res = self.module_list[i](x)
            x = nn.MaxPool1d(pool)(res)
            residuals_list.append(res)
        x = self.module_list[self.depth](x)
        residual = residuals_list[::-1]
        for i in range(0, self.depth*2, 2):
            up = nn.Upsample(scale_factor=2, mode='nearest')(x)
            x = self.decoder_list[i](up)
            merged = self.concat(x, residual[i//2])         
            x = self.decoder_list[i+1](merged)

        merged = self.match_x1_to_x2(x1=x, x2=input, value=0)
        pred = self.module_list[-1](merged)

        loss = 0
        acc = 0
        criterion = nn.CrossEntropyLoss()
        for i in range(len(y)):
            mask_idx = sum(y[i].ge(self.y_padtoken))
            masked_y = y[i][mask_idx:].unsqueeze(0)
            masked_pred = pred[i][:,mask_idx:].unsqueeze(0)
            loss += criterion(masked_pred, masked_y.long())
            acc += torch.sum(masked_pred.argmax(1) == masked_y, 1)/masked_y.shape[1]            

        loss /= y.shape[0]
        acc /= y.shape[0]
        return loss, acc, pred
