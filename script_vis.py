
import os

import torch
import numpy as np

from main import SegModel
from util import compute_performance

### Trained Model ###
#trained_root = './trained/lits/unet-baseline'
trained_root = './trained/lits/unet-w1-np64-nm4-bd'
model_path = os.path.join(trained_root, 'last.ckpt')
data_path = '/daintlab/data/segmentation/lits'
batch_size = 16
#####################

model = SegModel.load_from_checkpoint(model_path,
                                      data_path=data_path,
                                      batch_size=batch_size,
                                      num_layers=5,
                                      features_start=64)
model.cuda(device=0)
model.eval()

loader = model.test_dataloader()

fnames = []
dice_value = []
asd_value = []
acd_value = []
count = 1
for (inputs, labels, resized_labels, fname) in loader:
    bs = inputs.shape[0]
    inputs = inputs.cuda()
    labels = labels.cuda()
    resized_labels = resized_labels.cuda()
    outputs, embeddings = model.forward(inputs)
    metrics = compute_performance(outputs, labels, prefix='test', reduction='none')

    dice_value.extend(np.array(metrics['test_dice'].cpu())[:,0])
    asd_value.extend(np.array(metrics['test_asd'].cpu())[:,0])
    acd_value.extend(np.array(metrics['test_acd'].cpu())[:,0])
    fnames.extend(fname)

    if count % 50 == 0:
        print(f'[{count}/{len(loader)}] done')
    count += 1

dice = np.mean(dice_value)
asd = np.mean([i for i in asd_value if ~np.isnan(i)])
acd = np.mean([i for i in acd_value if ~np.isnan(i)])

print(f'Dice: {dice}, ASD: {asd}, ACD: {acd}')

