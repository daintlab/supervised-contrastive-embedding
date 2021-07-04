# Supervised Contrastive Embedding for Medical Image Segmentation
This repository provides an official pytorch lightning implementation of "Supervised Contrastive Embedding for Medical Image Segmentation".

## Framework Overview
![main_fig_v3](https://user-images.githubusercontent.com/62420967/124386765-e4a57600-dd16-11eb-80fc-0302c3939e33.jpg)


## Requirements
```
ubuntu 18.0.4, cuda 10.2
python 3.8
torch >= 1.7.1
torchvision >= 0.8.2
pytorch-lightning >= 1.1.8
monai
numpy
pillow
comet-ml (optional)
```

## Datasets
Following Datasets are used in our paper.
* Liver segmentation : LiTS Dataset - [Paper](https://arxiv.org/abs/1901.04056)
* Brain tumor segmentation : BraTS 2018 training dataset - [Paper](https://arxiv.org/abs/1811.02629)
* Lung segmentation : [JSRT](https://www.ajronline.org/doi/full/10.2214/ajr.174.1.1740071), [MC,SZ](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)
* Spinal cord segmentation : Spinal cord grey matter segmentation challenge - [Paper](https://www.sciencedirect.com/science/article/pii/S1053811917302185)

### Data preparation
Please refer to our paper for the preprocessing procedure of each dataset. Prepare the preprocessed dataset in the following format. Images and their corresponding labels(segmentation mask) must have the same file name. For source segmentation datasets(LiTS, BraTS), the train and test folders should contain two subfolders each containing images and labels. For domain generalization datasets(lung, spinal cord), each domain folders should contain four subfolders each. The image and label folders directly under the domain folder should contain all the images and labels of the domain respectively.
``` 
# For source segmentation dataset(LiTS, BraTS)

LiTS / train / image / 0.png
                     / 1.png
                     / ...
             / label / 0.png
                     / 1.png
                     / ...
     / test  / image / 0.png
                     / 1.png
                     / ...
             / label / 0.png
                     / 1.png
                     / ...
```
```
# For domain generalization dataset(Lung, spinal cord)

lung_seg / JSRT_dataset / image / 0.png
                                / 1.png
                                / ...
                        / label / 0.png
                                / 1.png
                                / ...
                        / train / image / 0.png
                                        / 1.png
                                        / ...
                                / label / 0.png
                                        / 1.png
                                        / ...
                        / test  / image / 0.png
                                        / 1.png
                                        / ...
                                / label / 0.png
                                        / 1.png
                                        / ...
         / MC_dataset   / ...
         / SZ_dataset   / ...
```

## Usage
Following training commands are examples of 4-gpus training. You can simply change ```--gpus``` to your available number of gpus.

### Train Baseline (Source segmentation)
Trained model will be saved in ```<Your save Path>```.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --data-path <Your Data Path> \
                                            --arch 'unet' --encoder 'resnet34' \
                                            --batch-size 16 --loss-weight 0 \
                                            --max-epochs 120 --default-root-dir <Your Save Path>
```

### Train SCE (Source segmentation)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --data-path <Your Data Path> \
                                            --arch 'unet' --encoder 'resnet34' \
                                            --batch-size 16 --loss-weight 1.0 \
                                            --max-epochs 120 --default-root-dir <Your Save Path> \
                                            --n-max-pos 128 --neg-multiplier 6
```

### Train SCE+linear (Source segmentation)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --data-path <Your Data Path> \
                                            --arch 'unet' --encoder 'resnet34' \
                                            --batch-size 16 --loss-weight 1.0 \
                                            --max-epochs 120 --default-root-dir <Your Save Path> \
                                            --n-max-pos 128 --neg-multiplier 6 \
                                            --boundary-aware --sampling-type 'linear'
```

### Train Baseline (Domain generalization)
Pass the path to each domain to the arguments. Data in ```data-path, source-data-path2, source-data-path3``` will be concatenated to form source dataset. 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --data-path <Your Data Path> \
                                            --arch 'unet' --encoder 'resnet50' \
                                            --batch-size 8 --loss-weight 0 \
                                            --max-epochs 120 --default-root-dir <Your Save Path> \
                                            --source-data-path2 <Your Data Path> \
                                            --source-data-path3 <Your Data Path>
```

### Evaluate Domain generalization performance
Pass the path to the target domain and the saved model to ```target-data-path``` and ```model-path``` respectively. Evaluation will be proceeded in single gpu and the performance will be saved in ```model-path```.
```
CUDA_VISIBLE_DEVICES=0 python domain_test.py --data-path <Your Data Path> \
                                             --source-data-path2 <Your Data Path> \
                                             --source-data-path3 <Your Data Path> \
                                             --target-data-path <Your Data Path> \
                                             --model-path <Path to your saved model>
                                             --arch 'unet' --encoder 'resnet50'
```


### Comet ML

You can monitor your training progress with comet ML by activating ```--logging``` argument. Make sure the following environment variables are set properly before you run the source code. Please modify workspace name in ```main.py```.
```
export COMET_API_KEY=<Your API Key> 
export COMET_PROJECT_NAME=<Your Project Name>
```

### Arguments
|Args|Description|Default|Type|
|----|-----------|-------|----|
|default-root-dir|Checkpoint save path|'./logs'|str|
|gpus|Number of gpus|8|int|
|logging|Whether to use Comet ML|-|action='store_true'|
|exp|Experiment name in Comet ML|'test'|str|
|data-path|Path to dataset|-|str|
|source-data-path2|Path to another source domain|-|str|
|source-data-path3|Path to another source domain|-|str|
|arch|Type of architecture|choices=['unet','unetpp','dlabv3','dlabv3p']|str|
|encoder|Type of encoder|choices=['resnet34','resnet50']|str|
|batch-size|Training batch size|32|int|
|lr|Learning rate|0.01|float|
|optim|Optimizer|'sgd'|str|
|loss-weight|Weight of contrastive loss|1.0|float|
|boundary-aware|Boundary-aware sampling|-|action='store_true'|
|sampling-type|Type of boundary-aware sampling|'fixed'|str|
|n-max-pos|Number of positive features to use|64|int|
|neg-multiplier|Multiplier to define number of negative features|6|int|
|max-epochs|Train epoch|120|int|


## Results

### Results on LiTS
|Method|Arch|Precision(%)|Recall(%)|Dice(%)|ACD|ASD|NLL|
|:----:|:--:|:----------:|:-------:|:-----:|:-:|:-:|:-:|
|Baseline|U-Net|88.37|85.41|85.43|6.30|6.96|0.099|
|SCE|U-Net|92.22|86.85|87.57|5.20|5.51|0.063|
|SCE+fixed|U-Net|92.10|87.38|87.90|5.02|5.31|0.062|
|SCE+random|U-Net|91.85|87.67|88.02|5.07|5.41|0.064|
|SCE+linear|U-Net|92.56|87.85|88.36|4.85|5.15|0.054|

### Results on Lung segmentation(MC,SZ->JSRT)
|   Method   |  Arch |  Source |      |       |  Target |      |       |
|:----------:|:-----:|:-------:|:----:|:-----:|:-------:|:----:|:-----:|
|            |       | Dice(%) |  ACD |  NLL  | Dice(%) |  ACD |  NLL  |
|  Baseline  | U-Net |  96.15  | 1.53 | 0.056 |  95.62  | 1.93 | 0.073 |
|     SCE    | U-Net |  96.21  |  1.5 | 0.054 |  96.05  | 1.73 | 0.063 |
| SCE+linear | U-Net |  96.18  | 1.52 | 0.055 |  96.19  | 1.68 |  0.06 |

More results can be found in the paper.
