
# Light-Head R-CNN Pytorch Implementation with Faster R-CNN based

  

## Introduction

  

This project is a *light-head R-CNN* pytorch implementation with faster R-CNN based, aimed to reducing the overhead of 'Head' part of faster R-CNN object detection models. My repository is based on following faster R-CNN version:

  

*  [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), developed based on pure PyTorch.

  

However, our implementation has several unique and new features compared with the above implementations (just like what jwyang did):

  

*  **It is pure Pytorch code**. We convert all the numpy implementations to pytorch!

  

*  **It supports multi-image batch training**. We revise all the layers, including dataloader, rpn, roi-pooling, etc., to support multiple images in each minibatch.

  

*  **It supports multiple GPUs training**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

  

*  **It supports three pooling methods**. We integrate three pooling methods: roi pooing, roi align and roi crop. More importantly, we modify all of them to support multi-image batch training.

  

*  **It is memory efficient**. We limit the image aspect ratio, and group images with similar aspect ratios into a minibatch. As such, we can train resnet101 and VGG16 with batchsize = 4 (4 images) on a sigle Titan X (12 GB). When training with 8 GPU, the maximum batchsize for each GPU is 3 (Res101), totally 24.

  

*  **It is faster**. Based on the above modifications, the training is much faster. We report the training speed on NVIDIA TITAN Xp in the tables below.

  

## Other Implementations

  

*  [Feature Pyramid Network (FPN)](https://github.com/jwyang/fpn.pytorch)

  

* Mask R-CNN (~~ongoing~~ already implemented by [roytseng-tw](https://github.com/roytseng-tw/mask-rcnn.pytorch))

  

## Tutorial

  

*  [Blog](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/) by [ankur6ue](https://github.com/ankur6ue)

  

## Important Notes

  

* I tested for pascal_voc 2007 dataset, working just fine. (As long as I know, [the original faster R-CNN code](https://github.com/jwyang/faster-rcnn.pytorch) works perfectly fine with other datasets, such as pascal 2012, COCO, etc.)
* However, this code __does not__ use PS RoI Pooling, yet. Therefore, benchmark result is expected to be slightly lower than the code that uses PS RoI Pooling.

  

### What we are going to do

  

- [x] Support both python2 and python3 (great thanks to [cclauss](https://github.com/cclauss)).

- [ ] ~~Add deformable pooling layer as an alternative way for roi pooling (mainly supported by [Xander](https://github.com/xanderchf))~~ Don't know when.

- [ ] ~~Run systematical experiments on PASCAL VOC 07/12, COCO, ImageNet, Visual Genome (VG) with different settings.~~

- [ ] ~~Write a detailed report about the new stuffs in our implementations, and the quantitative results in our experiments.~~

  

## Preparation

  
  

First of all, clone the code

```

git clone https://github.com/jungwon1413/pytorch-lighthead.git

```

  

Then, create a folder:

```

cd pytorch-lighthead && mkdir data

```

  

### prerequisites

  

* Python 2.7 or 3.6

* Pytorch 0.4.0 or higher

* CUDA 8.0 or higher is recommended

  

### Data Preparation

  

*  **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the folder data/.

  

*  **COCO**: Please also follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the data.

  

*  **Visual Genome**: Please follow the instructions in [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) to prepare Visual Genome dataset. You need to download the images and object annotation files first, and then perform proprecessing to obtain the vocabulary and cleansed annotations based on the scripts provided in this repository.

  

### Pretrained Model

  

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:

  

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

  

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

  

Download them and put them into the data/pretrained_model/.

  

**NOTE**. We compare the pretrained models from Pytorch and Caffe, and surprisingly find Caffe pretrained models have slightly better performance than Pytorch pretrained. We would suggest to use Caffe pretrained models from the above link to reproduce our results.

  

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

  

### Compilation

  

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code:

  

| GPU model | Architecture |

| ------------- | ------------- |

| TitanX (Maxwell/Pascal) | sm_52 |

| GTX 960M | sm_50 |

| GTX 1080 (Ti) | sm_61 |

| Grid K520 (AWS g2.2xlarge) | sm_30 |

| Tesla K80 (AWS p2.xlarge) | sm_37 |

  

More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

  

Install all the python dependencies using pip:

```

pip install -r requirements.txt

```

  

Compile the cuda dependencies using following simple commands:

  

```

cd lib

sh make.sh

```

  

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

  

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

  

## Train

  

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

  

To train a faster R-CNN model with vgg16 on pascal_voc, simply run:

```

CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \

--dataset pascal_voc --net vgg16 \

--bs $BATCH_SIZE --nw $WORKER_NUMBER \

--lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \

--cuda --lighthead

```

where 'bs' is the batch size with default 1. Alternatively, to train with resnet101 on pascal_voc, simple run:

```

CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \

--dataset pascal_voc --net res101 \

--bs $BATCH_SIZE --nw $WORKER_NUMBER \

--lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \

--cuda --lighthead

```

Above, BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size. **On Titan Xp with 12G memory, it can be up to 4**.

  

If you have multiple (say 8) Titan Xp GPUs, then just use them all! Try:

```

python trainval_net.py --dataset pascal_voc --net vgg16 \

--bs 24 --nw 8 \

--lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \

--cuda --mGPUs --lighthead

  

```

  

Change dataset to "coco" or 'vg' if you want to train on COCO or Visual Genome.

  

## Test

  

If you want to evlauate the detection performance of a pre-trained vgg16 model on pascal_voc test set, simply run

```

python test_net.py --dataset pascal_voc --net vgg16 \

--checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \

--cuda --lighthead

```

Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.

  

## Demo

  

If you want to run detection on your own images with a pre-trained model, download the pretrained model listed in above tables or train your own models at first, then add images to folder $ROOT/images, and then run

```

python demo.py --net vgg16 \

--checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \

--cuda --lighthead --load_dir path/to/model/directoy

```

  

Then you will find the detection results in folder $ROOT/images.

  

**Note the default demo.py merely support pascal_voc categories. You need to change the [line](https://github.com/jwyang/faster-rcnn.pytorch/blob/530f3fdccaa60d05fa068bc2148695211586bd88/demo.py#L156) to adapt your own model.**

  

Below are some detection results:

  

<div  style="color:#0000FF"  align="center">

<img  src="images/img3_det_res101.jpg"  width="430"/>  <img  src="images/img4_det_res101.jpg"  width="430"/>

</div>

However, there would be some errors on demo file since I did not work on yet.
  

## Webcam Demo

  

You can use a webcam in a real-time demo by running

```

python demo.py --net vgg16 \

--checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \

--cuda --load_dir path/to/model/directoy \

--webcam $WEBCAM_ID

```

The demo is stopped by clicking the image window and then pressing the 'q' key.



  

## Citation

  

@article{jjfaster2rcnn,

Author = {Zeming Li , Chao Peng , Gang Yu , Xiangyu Zhang , Yangdong Deng , Jian Sun},

Title = {Light-Head R-CNN: In Defense of Two-Stage Object Detector},

Journal = {https://github.com/zengarden/light_head_rcnn},

Year = {2017}

}