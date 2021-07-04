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
* Liver segmentation : LiTS Dataset
* Brain tumor segmentation : BraTS 2018 training dataset
* Lung segmentation : JSRT, MC, SZ
* Spinal cord segmentation : Spinal cord grey matter segmentation challenge

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

### Train Baseline (Domain generalization : spinal cord)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --gpus 4 --data-path <Your Data Path> \
                                            --arch 'unet' --encoder 'resnet50' \
                                            --batch-size 8 --loss-weight 0 \
                                            --max-epochs 120 --default-root-dir <Your Save Path> \
                                            --source-data-path2 <Your Data Path> \
                                            --source-data-path3 <Your Data Path>
```

### Evaluate Domain generalization performance
Evaluation will be proceeded in single gpu.
```
CUDA_VISIBLE_DEVICES=0 python domain_test.py --data-path <Your Data Path> \
                                             --source-data-path2 <Your Data Path> \
                                             --source-data-path3 <Your Data Path> \
                                             --target-data-path <Your Data Path> \
                                             --model-path <Path to your saved model>
                                             --arch 'unet' --encoder 'resnet50'
```


### Comet ML

You can monitor your training progress with comet ML by activating ```--logging``` argument. Make sure 
