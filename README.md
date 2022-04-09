# DCGAN  
[paper](https://arxiv.org/pdf/1511.06434.pdf)
Pytorch implementation.

# 1. Dataset  
I downloaded dataset from [kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)  

The dataset has 202,599 images and sizes are same 218 × 178.

Zip file from kaggle, put the .zip in the folder `face` then use `data_unzip.py` unzip it.  

In paper, image size are resized as 64 × 64.

In `dataloader.py`, I made custom Dataset and Transformations.  
And It allows you to resize.  

After training with 64 size,  
I tried with higher resolution images 160 × 160 with model maintained.  

To do so, some variations in codes are needed.  

First of all, one need to check how the output do Discriminator for the size.  
For 160 × 160, output tensor of Discriminator is [100 × 7 × 7].  

# 2. Train.  
## 2.1 64 size
`python train.py --face_dir /home/mskang/hyeokjong/GAN/face/images/img_align_celeba --num_epochs 50 --lr 0.0002 --batch_size 512 --result_dir /home/mskang/hyeokjong/GAN/result --device cuda:1`

## 2.2 160 size with same model  
`python train.py --face_dir /home/mskang/hyeokjong/GAN/face/images/img_align_celeba --num_epochs 15 --lr 0.0002 --batch_size 256 --result_dir /home/mskang/hyeokjong/GAN/result --device cuda:1`  
For this, one should change codes in `dataloader.py`, `trainer.py`.  That' all.  
I added annotations for those files with emoji 🤓.


## 2.3  128 size and add more convs and transconvs.  
For this one need to change codes in `dataloader.py`, `generator.py`, `discriminator.py`.  
I added annotations for those files with emoji 🦢.  




# 3. Results.  

After train, one can get images which are generated by Generator every 300 iteration.  

And Using `GIF` one can get sequence of images generated. 

`python make_GIF.py --images_dir /home/mskang/hyeokjong/GAN/result/images --GIF_dir /home/mskang/hyeokjong/GAN/result/GIF`

`generate.py` allows to make new image by trained Generator.
`python generate.py --pt_dir /home/mskang/hyeokjong/GAN/result/trained_generators.pt --save_dir /home/mskang/hyeokjong/GAN/result/images --GPU cuda:0`  

Check how my images generated.  

## 3.1 64 size  

## 3.2 160 size with same model  

## 3.3 128 size and add more convs and transconvs.  

