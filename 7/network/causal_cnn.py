# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Implementation of causal CNNs partly taken and modified from
# https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries.


import torch
import torch.nn as nn


class Softplus(nn.Module):
    def __init__(self, eps):
        super(Softplus, self).__init__()
        self.eps = eps
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x) + self.eps


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param padding Zero-padding applied to the left of the input of the
           non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False, forward=True):
        super(CausalConvolutionBlock, self).__init__()

        Conv1d = torch.nn.Conv1d if forward else torch.nn.ConvTranspose1d
        
        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size, forward=True):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        # double the dilation size if forward, if backward
        # we start at the final dilation and work backwards
        dilation_size = 1 if forward else 2**depth
        
        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size,
                forward,
            )]
            # double the dilation at each step if forward, otherwise
            # halve the dilation
            dilation_size = dilation_size*2 if forward else dilation_size//2

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Spatial(nn.Module):
  def __init__(self, channels, dropout, forward=True):
    super(Spatial, self).__init__()
    Conv1d = nn.Conv1d if forward else nn.ConvTranspose1d
    self.network = nn.Sequential(
      Conv1d(channels, channels, 1),
      nn.BatchNorm1d(num_features=channels),
      nn.ReLU(),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size, dropout):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        spatial = Spatial(reduced_size, dropout)
        reduce_size = torch.nn.AdaptiveAvgPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear1 = torch.nn.Linear(reduced_size, 26)
        linear2 = torch.nn.Linear(26, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, spatial, reduce_size, squeeze, linear1,
            nn.BatchNorm1d(num_features=26), nn.ReLU(), nn.Dropout(dropout), linear2,
        )

    def forward(self, x):
        return self.network(x)
      
      
class CausalCNNVEncoder(torch.nn.Module):
    """
    Variational encoder. Difference is that we need two outputs: mean and
    standard deviation.
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size, softplus_eps, dropout):
        super(CausalCNNVEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze,
        )
        self.linear_mean = torch.nn.Linear(reduced_size, out_channels)
        self.linear_sd = torch.nn.Sequential(
            torch.nn.Linear(reduced_size, out_channels),
            Softplus(softplus_eps),
        )

    def forward(self, x):
        out = self.network(x)
        return self.linear_mean(out), self.linear_sd(out)

      
class CausalCNNVDecoder(torch.nn.Module):
    """
    Variational decoder.
    """
    def __init__(self, k, width, in_channels, channels, depth, out_channels,
               kernel_size, gaussian_out, softplus_eps, dropout):
        super(CausalCNNVDecoder, self).__init__()
        self.in_channels = in_channels
        self.width = width
        self.gaussian_out = gaussian_out
        self.linear1 = torch.nn.Linear(k, in_channels)
        self.linear2 = torch.nn.Linear(in_channels, in_channels*width)
        self.causal_cnn = CausalCNN(
            in_channels, channels, depth, out_channels, kernel_size,
            forward=False,
        )
        if self.gaussian_out:
            self.linear_mean = nn.Linear(out_channels*width, out_channels*width)
            self.linear_sd = torch.nn.Sequential(
                torch.nn.Linear(out_channels*width, out_channels*width),
                Softplus(softplus_eps),
            )
        
    def forward(self, x):
        """
        Returns a reconstruction of the original 8x600 ECG, by decoding
        the given compression.
        """
        B, _ = x.shape
        # from (BxK) to (BxC)
        out = self.linear1(x)
        # from (BxC) to (Bx(C*600))
        out = self.linear2(out)
        # from (Bx(C*600)) to (BxCx600)
        out = out.view(B, self.in_channels, self.width)
        # deconvolve through the causal CNN
        out = self.causal_cnn(out)
        if self.gaussian_out:
          shape = out.shape
          # flatten the output to shape (Bx(8*600))
          out = torch.flatten(out, start_dim=1)
          return self.linear_mean(out), self.linear_sd(out)
        return out
