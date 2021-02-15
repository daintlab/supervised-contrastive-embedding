import torch
import numpy as np
from monai import metrics

def compute_performance(output, label, prefix=None, reduction='mean'):
    #output = (output >= 0).cpu().detach().numpy()
    binary_output = output > 0.0

    # dice
    dice, _ = metrics.DiceMetric(reduction=reduction)(binary_output, label)

    # hausdorff distance
    #hausdorff_value, _ = metrics.HausdorffDistanceMetric()(output, label)
    #hausdorff_value = hausdorff_value.cuda()

    # average surface distance
    #asd_value, _ = metrics.SurfaceDistanceMetric(symmetric=True,
    #                                             include_background=True)(output, label)
    #asd_value = asd_value.cuda()

    #return {'dice': dice_value,
    #        'asd': asd_value}


    #probs = torch.sigmoid(output)
    #output = (probs > 0.5).float()
    #probs = probs.cpu().numpy().squeeze()
    #label = label.cpu().numpy().squeeze()

    #output = output > 0.0

    batch_size, n_channel = output.shape[:2]

    asd = np.empty((batch_size, n_channel))
    acd = np.empty((batch_size, n_channel))

    for b, c in np.ndindex(batch_size, n_channel):
        edges_pred, edges_gt = metrics.utils.get_mask_edges(binary_output[b,c],
                                                            label[b,c])
        asd_1 = metrics.utils.get_surface_distance(edges_pred, edges_gt)
        asd_2 = metrics.utils.get_surface_distance(edges_gt, edges_pred)

        if binary_output[b,c].sum() == 0: # failed to predict
            asd[b,c] = np.nan
            acd[b,c] = np.nan
        else:
            asd[b,c] = (asd_1.sum() + asd_2.sum()) / (len(asd_1) + len(asd_2))
            acd[b,c] = ((asd_1.sum()/len(asd_1)) + (asd_2.sum()/len(asd_2))) / 2

    if reduction == 'mean':
        asd = asd[~np.isnan(asd)[:,0],:].mean()
        acd = acd[~np.isnan(acd)[:,0],:].mean()

    dice = dice.type_as(output)
    asd = torch.from_numpy(np.array(asd)).type_as(output)
    acd = torch.from_numpy(np.array(acd)).type_as(output)

    result = {'dice': dice, 'asd': asd, 'acd': acd}

    if prefix:
        result = {prefix+'_'+key:value for key, value in result.items()}

    return result
