# GCN-Based-Interactive-Prostate-Segmentationon-on-MR-Images
PyTorch code for Graph convolutional network based interactive prostate segmentation on MR images 

## Note
The trained interactive model file epoch34_step14314.pth and pretrained [model file](https://drive.google.com/drive/my-drive?ths=true) MS_DeepLab_resnet_pretained_VOC.pth can be download.


## Segmentation Result
<img src = "GCN-Based-Interactive-Prostate-Segmentationon-on-MR-Images/doc/segmentation result.PNG" width="56%"/>

## Train
- Run the run_train.sh shell script with,
```
sh run_train.sh
```

## Test
- Run the run_prediction.sh shell script with,
```
sh run_prediction.sh
```