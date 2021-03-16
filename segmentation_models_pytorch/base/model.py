import torch
from . import initialization as init
from . import modules as md

import torch.nn as nn
import torch.nn.functional as F

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

        #self.proj_conv1 = md.Conv2dReLU(512, 256, kernel_size=1, padding=1, use_batchnorm=False)
        #self.proj_conv2 = md.Conv2dReLU(256, 128, kernel_size=1, padding=1, use_batchnorm=False)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        #embedding = self.proj_conv1(features[-1])
        #embedding = self.proj_conv2(embedding)
        #embedding = F.interpolate(embedding, size=128, mode='bilinear', align_corners=False)
        embedding = F.interpolate(features[-1], size=128, mode='bilinear', align_corners=False)

        return masks, embedding

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
