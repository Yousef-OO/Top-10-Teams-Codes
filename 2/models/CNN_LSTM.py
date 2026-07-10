import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv3x1(in_planes, out_planes):
    """3x1 convolution maintains the length"""
    conv3x1same = nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True))
    return conv3x1same

def conv24x1(in_planes, out_planes):
    """24x1 convolution maintains the length / 2"""
    conv24x1same = nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=24, stride=2, padding=11, bias=False),
        nn.LeakyReLU(0.2, inplace=True))
    return conv24x1same


def conv48x1(in_planes, out_planes):
    """48x1 convolution maintains the length / 2"""
    conv24x1same = nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=48, stride=2, padding=23, bias=False),
        nn.LeakyReLU(0.2, inplace=True))
    return conv24x1same


class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, inplanes)
        self.conv2 = conv24x1(inplanes, outplanes)
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.dropout(out)

        return out

# For summarizing a set of vectors into a single vector
class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size, dropout=0.):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x = [batch, len, hdim]
        alpha = [batch,sentence_len]
        output = [batch,hidden_dim]
        """
        x = self.dropout(x)

        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        alpha = F.softmax(scores, dim=1)
        output = alpha.unsqueeze(1).bmm(x).squeeze(1)
        return output, alpha


class CNNLSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(CNNLSTM, self).__init__()

        self.embed1 = self._make_layer(in_channel, in_channel, number=4)
        self.embed2 = nn.Sequential(
            conv3x1(in_channel, in_channel),
            conv3x1(in_channel, in_channel),
            conv48x1(in_channel, in_channel),
            nn.Dropout(.2)
            )
        self.bigru = nn.GRU(12, 12, num_layers=1,
                            bidirectional=True, batch_first=True, bias=False)

        self.attention = LinearSelfAttn(24)

        self.lrelu = nn.LeakyReLU(0.3, inplace=True)
        self.dropout = nn.Dropout(.2)
        self.activation = nn.Sequential(
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(.2),
            )
        self.fc = nn.Linear(24*2250, out_channel)

    def _make_layer(self, inplanes, outplanes, number):
        layers = []
        for _ in range(0, number):
            layers.append(BasicBlock(inplanes, outplanes))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.embed1(x)
        x = self.embed2(x)
        x = torch.transpose(x, 1, 2)
        bigru_out, _ = self.bigru(x)
        #x,_ = self.attention(bigru_out)
        x = torch.transpose(bigru_out, 1, 2)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        return x
