
import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import distributed
from torchsummary import summary

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

pl.utilities.seed.seed_everything(seed=1234)

from unet import UNet
from dataset import OrganData
from losses import DiceLoss, PixelwiseContrastiveLoss
from util import compute_performance


class SegModel(pl.LightningModule):

    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 lr: float = 0.01,
                 loss_weight: float = 0.1,
                 n_max_pos: int = 128,
                 boundary_aware: bool = False,
                 sampling_type: str = 'full',
                 neg_multiplier: int = 1,
                 num_layers: int = 4,
                 features_start: int = 32,
                 use_ddp: bool = False,
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr
        self.loss_weight = loss_weight
        self.n_max_pos = n_max_pos
        self.boundary_aware = boundary_aware
        self.sampling_type = sampling_type
        self.neg_multiplier = neg_multiplier
        self.num_layers = num_layers
        self.features_start = features_start
        self.use_ddp = use_ddp
        self.max_epochs = kwargs['max_epochs']

        self.net = UNet(num_classes=1, num_layers=self.num_layers,
                        features_start=self.features_start)

        self.train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])])

        self.trainset = OrganData(data_path=os.path.join(self.data_path, 'train'),
                                  transform=self.train_transform)
        self.testset = OrganData(data_path=os.path.join(self.data_path, 'test'),
                                 transform=self.val_transform)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.cont_loss = PixelwiseContrastiveLoss(neg_multiplier=self.neg_multiplier,
                                                  n_max_pos=self.n_max_pos,
                                                  boundary_aware=self.boundary_aware,
                                                  sampling_type=self.sampling_type)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, label, resized_label = batch
        img = img.float()

        output, embedding = self(img)

        cont_loss = self.cont_loss(embedding, resized_label,
                                   split_param=(self.current_epoch, self.max_epochs))
        loss = self.bce_loss(output, label) + \
               self.dice_loss(output, label) + \
               self.loss_weight * cont_loss

        self.log('train_cont_loss', cont_loss,
                 prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_loss', loss,
                 prog_bar=True, on_step=True, on_epoch=True)
        # metrics
        metric_log = compute_performance(output, label, prefix='train')
        #import ipdb; ipdb.set_trace()
        self.log_dict(metric_log,
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label, _ = batch
        img = img.float()

        output, _ = self(img)
        loss = self.bce_loss(output, label) + self.dice_loss(output, label)

        self.log('val_loss', loss,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # metrics
        metric_log = compute_performance(output, label, prefix='val')
        #import ipdb; ipdb.set_trace()
        self.log_dict(metric_log,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.net.parameters(),
                              lr=self.lr,
                              momentum=0.9,
                              weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                         milestones=[80,100],
                                                         gamma=0.1)
        return [opt], [scheduler]

    def train_dataloader(self):
        if self.use_ddp:
            train_sampler = distributed.DistributedSampler(self.trainset)
        else:
            train_sampler = None
        return DataLoader(self.trainset, batch_size=self.batch_size,
                          shuffle=(train_sampler is None),
                          num_workers=4,
                          sampler=train_sampler,
                          pin_memory=True)

    def val_dataloader(self):
        if self.use_ddp:
            val_sampler = distributed.DistributedSampler(self.testset)
        else:
            val_sampler = None
        return DataLoader(self.testset, batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4,
                          sampler=val_sampler,
                          pin_memory=True)

    def test_dataloader(self):
        self.testset = OrganData(data_path=os.path.join(self.data_path, 'test'),
                                 transform=self.val_transform, mode='test')

        if self.use_ddp:
            test_sampler = distributed.DistributedSampler(self.testset)
        else:
            test_sampler = None
        return DataLoader(self.testset, batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4,
                          sampler=test_sampler,
                          pin_memory=True)



    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--data-path", type=str, default='/daintlab/data/segmentation/lits')
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--loss-weight", type=float, default=1.0)
        parser.add_argument("--boundary-aware", action='store_true')
        parser.add_argument("--sampling-type", type=str, default='full')
        parser.add_argument("--n-max-pos", type=int, default=128)
        parser.add_argument("--neg-multiplier", type=int, default=1)
        parser.add_argument("--num-layers", type=int, default=5)
        parser.add_argument("--features-start", type=int, default=64)
        parser.add_argument("--max-epochs", type=int, default=110)
        parser.add_argument("--deterministic", type=bool, default=True)

        return parser


def main(hparams):

    if hparams.logging:
        logger = CometLogger(api_key=os.environ.get('COMET_API_KEY'),
                            workspace='beopst',
                            project_name='seg-with-cont',
                            experiment_name=hparams.exp)
        hparams.logger = logger
        hparams.logger.log_hyperparams(hparams)

    ckpt_callback = ModelCheckpoint(save_last=True)
    hparams.callbacks = [ckpt_callback]

    hparams.sync_batchnorm = True

    model = SegModel(**vars(hparams))

    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(model)

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--default-root-dir", type=str, default='./logs')
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--replace-sampler-ddp", type=bool, default=False)
    parser.add_argument("--use-ddp", type=bool, default=True)
    parser.add_argument("--accelerator", type=str, default='ddp')
    parser.add_argument("--exp", type=str, default='test')
    parser.add_argument("--logging", action='store_true')
    parser = SegModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)
