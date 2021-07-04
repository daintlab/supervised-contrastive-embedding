
import os
import torch
import numpy as np
from argparse import ArgumentParser
import torch.nn.functional as F
import json

from main import SegModel
from util import compute_performance

def eval(model_path,arch,encoder,data_path,**kwargs):
    ### Loading Trained Model ###
    save_path = os.path.dirname(model_path)
    batch_size = 16
    model = SegModel.load_from_checkpoint(model_path,
                                        data_path=data_path,
                                        batch_size=batch_size,
                                        num_layers=5,
                                        arch=arch,
                                        encoder=encoder,
                                        features_start=64,
                                        **kwargs)

    model.cuda(device=0)
    model.eval()

    # Dataloader
    source_loader = model.test_dataloader()
    target_loader = model.target_test_dataloader()

    loader_list = [source_loader, target_loader]
    domain_list = ["source","target"]

    perform_dict={}

    for idx in range(len(loader_list)):
        loader = loader_list[idx]
        domain = domain_list[idx]

        fnames = []
        recall_value = []
        precision_value = []
        f1_value = []
        dice_value = []
        asd_value = []
        acd_value = []
        loss_value = []
        count = 1
        with torch.no_grad():
            for (inputs, labels, resized_labels, fname) in loader:
                bs = inputs.shape[0]
                inputs = inputs.cuda()
                labels = labels.cuda()
                resized_labels = resized_labels.cuda()
                outputs, embeddings = model.forward(inputs)
                loss = model.bce_loss(outputs,labels)

                metrics = compute_performance(outputs, labels,
                                            metric=['confusion','dice','asd','acd'],
                                            prefix=f'test_{domain}', reduction='none')

                recall = np.array(metrics[f'test_{domain}_recall'].cpu())[:,0]
                precision = np.array(metrics[f'test_{domain}_precision'].cpu())[:,0]
                dice = np.array(metrics[f'test_{domain}_dice'].cpu())[:,0]
                asd = np.array(metrics[f'test_{domain}_asd'].cpu())[:,0]
                acd = np.array(metrics[f'test_{domain}_acd'].cpu())[:,0]

                recall_value.extend(recall)
                precision_value.extend(precision)
                dice_value.extend(dice)
                asd_value.extend(asd)
                acd_value.extend(acd)
                loss_value.append(loss.cpu().item())
                fnames.extend(fname)

                print(f'[{count*batch_size}/{len(loader.dataset)}] done')
                count += 1

        recall = np.mean(recall_value)
        precision = np.mean(precision_value)
        dice = np.mean(dice_value)
        asd = np.mean([i for i in asd_value if ~np.isnan(i)])
        acd = np.mean([i for i in acd_value if ~np.isnan(i)])
        loss = np.mean(loss_value)

        print(f'Test_{domain} Precision : {precision}, Recall : {recall}, Dice: {dice}, ASD: {asd}, ACD: {acd}, Loss : {loss}')
        perform_dict[f'precision_{domain}'] = precision
        perform_dict[f'recall_{domain}'] = recall
        perform_dict[f'dice_{domain}'] = dice
        perform_dict[f'asd_{domain}'] = asd
        perform_dict[f'acd_{domain}'] = acd
        perform_dict[f'loss_{domain}'] = loss

    perform_dict={k:float(v) for k,v in perform_dict.items()}
    with open(os.path.join(save_path,"performance.json"),"w") as fp:
        json.dump(perform_dict,fp,indent=2)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--data-path", type=str, default='./logs')
    parser.add_argument("--source-data-path2", type=str, default=None)
    parser.add_argument("--source-data-path3", type=str, default=None)
    parser.add_argument("--target-data-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--arch", type=str, default='unet')
    parser.add_argument("--encoder", type=str, default='resnet50')
    hparams = parser.parse_args()
    
    eval(model_path=hparams.model_path, arch=hparams.arch, encoder=hparams.encoder,
        source_data_path2=hparams.source_data_path2,
        source_data_path3=hparams.source_data_path3,
        target_data_path=hparams.target_data_path)



