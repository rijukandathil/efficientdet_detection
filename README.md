# Detection of Cars and Person Using Efficientdet

#### Install requirements
```
pip3 install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
pip3 install torch==1.4.0
pip3 install torchvision==0.5.0
```

#### Clone the repository
```
git clone https://github.com/rijukandathil/efficientdet_detection.git
cd efficientdet_detection
```
#### Download the dataset and unzip the folder

```
wget https://evp-ml-data.s3.us-east-2.amazonaws.com/ml-interview/openimages-personcar/trainval.tar.gz
tar -xvzf trainval.tar.gz
```
#### Dataset preparation 

```
# dataset structure should be like this
datasets/
    -your_project_name/
        -train_set_name/
            -*.jpg
        -val_set_name/
            -*.jpg
        -annotations
            -instances_{train_set_name}.json
            -instances_{val_set_name}.json
```
#### Download pretrained weights

Here we use [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth) and saved under `weights` folder.

| coefficient | pth_download | GPU Mem(MB) | FPS | mAP 0.5:0.9(official) |
| ----------- | ------------ | ----------- | --- | --------------------- |
| D2 | [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth) | 1321 | 26.50 | 43.0 |

####  Manual set project's specific parameters
Create a `(efficientdet.yml)` file  under `projects` folder 

```
project_name: efficientdet 
train_set: train
val_set: val
num_gpus: 1

mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]

anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
obj_list: ['person', 'car']
```

#### Train a custom dataset with pretrained weights

```
python3 train.py -c 2 -p efficientdet --batch_size 1 --lr 1e-3 --num_epochs 20 \
 --load_weights /weights/efficientdet-d2.pth
```

The trained weights are saved under `logs` folder. It can be downloaded from [Google Drive](https://drive.google.com/file/d/1EIsfyam9HglARNsWEP0_wNKNnZmP8udB/view?usp=sharing)

```
# to resume training from the last checkpoint
python3 train.py -c 2 -p efficientdet --batch_size 1 --lr 1e-3 --num_epochs 20 \
 --load_weights last
```
#### Testing the model on a custom image

Modify the efficientdet_test.py and run it.
`python3 efficientdet_test.py`


![Sample output](https://i.imgur.com/1rFudt3.jpg)

