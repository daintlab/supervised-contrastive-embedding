import torch
import numpy as np
from monai import metrics

def compute_performance(output, label, metric, prefix=None, reduction='mean'):

    binary_output = output > 0.0

    result = {}
    if 'confusion' in metric:
        confusion = metrics.ConfusionMatrixMetric(metric_name=['recall','precision','f1_score'],
                                                  compute_sample=True,
                                                  reduction=reduction)(binary_output,label)
        recall = confusion[0]
        precision = confusion[2]
        f1_score = confusion[4]

        result['recall'] = recall
        result['precision'] = precision
        result['f1'] = f1_score


    if 'dice' in metric:
        # dice
        dice, _ = metrics.DiceMetric(reduction=reduction)(binary_output, label)
        dice = dice.type_as(output)
        result['dice'] = dice

    if ('asd' in metric) or ('acd' in metric):

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

        asd = torch.from_numpy(np.array(asd)).type_as(output)
        acd = torch.from_numpy(np.array(acd)).type_as(output)

        result['asd'] = asd
        result['acd'] = acd

    if prefix:
        result = {prefix+'_'+key:value for key, value in result.items()}

    return result
