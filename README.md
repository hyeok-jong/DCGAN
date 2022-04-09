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

By training with 64 size, I got  

<img src='https://velog.velcdn.com/images/jj770206/post/8440b4a0-20fa-4bc4-83c5-56e879246a8f/11th.gif'  style="width: 300px; height: auto;">   

Then I tried to train with higher resolution images 160 × 160 with model maintained.  
And I got  

<img src=''  style="width: 300px; height: auto;">

To do so, some variations in codes are needed.  

First of all, u need to check how the output do Discriminator for the size.  
For 160 × 160, output tensor of Discriminator is [100 × 7 × 7].  

Then one should change in 'dataloader.py', 'trainer.py'.  That' all.
I add commitions for those file with emoji 🤓.

# 2. Train.  
`python train.py --face_dir /home/mskang/hyeokjong/GAN/face/images/img_align_celeba --num_epochs 10 --lr 0.0002 --batch_size 512 --result_dir /home/mskang/hyeokjong/GAN/result --device cuda:1`

# 3. Results.  

After train, one can get images which are generated by Generator every 300 iteration.  

And Using `GIF` one can get sequence of images generated. 

`python make_GIF.py --images_dir /home/mskang/hyeokjong/GAN/result/images --GIF_dir /home/mskang/hyeokjong/GAN/result/GIF`

`generate.py` allows to make new image by trained Generator.
`python generate.py --pt_dir /home/mskang/hyeokjong/GAN/result/trained_generators.pt --save_dir /home/mskang/hyeokjong/GAN/result/images --GPU cuda:0`
