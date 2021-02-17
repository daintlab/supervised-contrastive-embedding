
import os

import torch
import numpy as np
np.random.seed(123)

from main import SegModel
from util import compute_performance

import torch.nn.functional as F

### Trained Model ###
#trained_root = './trained/lits/unet-baseline'
#trained_root = './trained/lits/unet-w1-np64-nm4-bd-random'
trained_root = './trained/lits/unet-w1-np64-nm4-bd-linear'
model_path = os.path.join(trained_root, 'last.ckpt')
data_path = '/daintlab/data/segmentation/lits'
batch_size = 4
#####################

def extract_boundary(label, is_pos=True):
    if not is_pos:
        label = 1 - label

    gt_b = F.max_pool2d(1-label, kernel_size=3, stride=1, padding=1)
    gt_b_in = 1 - gt_b
    gt_b -= 1 - label
    return gt_b, gt_b_in

def sample_balance(label, n):
    cand_pixels = torch.nonzero(label)
    batch_idx = cand_pixels[:,0]
    bs = batch_idx.max()
    sample_idx = []
    accum = 0
    for b in range(bs+1):
        n_features = int((batch_idx == b).sum().cpu())
        temp_idx = np.random.permutation(n_features)[:n] + accum
        sample_idx += temp_idx.tolist()
        accum += n_features

    sample_pixels = tuple(cand_pixels[sample_idx].t())

    return sample_pixels


def split_features(embedding, label, n=1):

    # extract boundary
    pos_b, pos_b_in = extract_boundary(label)
    neg_b, neg_b_in = extract_boundary(label, is_pos=False)

    try:
        pos_b_pixels = sample_balance(pos_b, n)
        pos_b_in_pixels = sample_balance(pos_b_in, n)
        neg_b_pixels = sample_balance(neg_b, n)
        neg_b_in_pixels = sample_balance(neg_b_in, n)

        pos_b_features = embedding[pos_b_pixels[0],:,pos_b_pixels[2],pos_b_pixels[3]]
        pos_b_in_features = embedding[pos_b_in_pixels[0],:,pos_b_in_pixels[2],pos_b_in_pixels[3]]
        neg_b_features = embedding[neg_b_pixels[0],:,neg_b_pixels[2],neg_b_pixels[3]]
        neg_b_in_features = embedding[neg_b_in_pixels[0],:,neg_b_in_pixels[2],neg_b_in_pixels[3]]
    except:
        return None, None, None, None

    #sample_idx = np.random.permutation(pos_features.shape[0])[:batch_size*n]
    #pos_features = pos_features[sample_idx]

    #sample_idx = np.random.permutation(neg_features.shape[0])[:batch_size*n]
    #neg_features = neg_features[sample_idx]

    return pos_b_features, pos_b_in_features, neg_b_features, neg_b_in_features

model = SegModel.load_from_checkpoint(model_path,
                                      data_path=data_path,
                                      batch_size=batch_size,
                                      num_layers=5,
                                      features_start=64)
model.cuda(device=0)
model.eval()

loader = model.test_dataloader()

fnames = []
recall_value = []
precision_value = []
f1_value = []
dice_value = []
asd_value = []
acd_value = []
pos_b_features = []
pos_b_in_features = []
neg_b_features = []
neg_b_in_features = []
count = 1
for (inputs, labels, resized_labels, fname) in loader:
    bs = inputs.shape[0]
    inputs = inputs.cuda()
    labels = labels.cuda()
    resized_labels = resized_labels.cuda()
    outputs, embeddings = model.forward(inputs)
    metrics = compute_performance(outputs, labels,
                                  metric=['confusion','dice','asd'],
                                  prefix='test', reduction='none')

    recall_value.extend(np.array(metrics['test_recall'].cpu())[:,0])
    precision_value.extend(np.array(metrics['test_precision'].cpu())[:,0])
    f1_value.extend(np.array(metrics['test_f1'].cpu())[:,0])
    dice_value.extend(np.array(metrics['test_dice'].cpu())[:,0])
    asd_value.extend(np.array(metrics['test_asd'].cpu())[:,0])
    acd_value.extend(np.array(metrics['test_acd'].cpu())[:,0])
    fnames.extend(fname)

    pos_b_feature, pos_b_in_feature, neg_b_feature, neg_b_in_feature = split_features(embeddings, resized_labels)

    if pos_b_feature is not None:
        pos_b_feature = pos_b_feature.detach().cpu()
        pos_b_in_feature = pos_b_in_feature.detach().cpu()
        neg_b_feature = neg_b_feature.detach().cpu()
        neg_b_in_feature = neg_b_in_feature.detach().cpu()

        pos_b_features.append(pos_b_feature)
        pos_b_in_features.append(pos_b_in_feature)
        neg_b_features.append(neg_b_feature)
        neg_b_in_features.append(neg_b_in_feature)


    if count % 50 == 0:
        print(f'[{count}/{len(loader)}] done')
    count += 1

recall = np.mean(recall_value)
precision = np.mean(precision_value)
f1 = np.mean(f1_value)
dice = np.mean(dice_value)
asd = np.mean([i for i in asd_value if ~np.isnan(i)])
acd = np.mean([i for i in acd_value if ~np.isnan(i)])

print(f'Recall: {recall}, Precision: {precision}, F1: {f1}')
print(f'Dice: {dice}, ASD: {asd}, ACD: {acd}')

print(recall,precision,f1,dice,asd,acd)

pos_b_features = np.concatenate(pos_b_features)
np.save(os.path.join(trained_root, 'pos_b_features.npy'), pos_b_features)
pos_b_in_features = np.concatenate(pos_b_in_features)
np.save(os.path.join(trained_root, 'pos_b_in_features.npy'), pos_b_in_features)

neg_b_features = np.concatenate(neg_b_features)
np.save(os.path.join(trained_root, 'neg_b_features.npy'), neg_b_features)
neg_b_in_features = np.concatenate(neg_b_in_features)
np.save(os.path.join(trained_root, 'neg_b_in_features.npy'), neg_b_in_features)
