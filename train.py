from dataloader import make_dataloader
from discriminator import Discriminator
from generator import Generator
from trainer import trainer
from loss import loss_function
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face_dir", type = str, help = "image directory")
    parser.add_argument("--num_epochs", type = int, help = "image directory")
    parser.add_argument("--lr", type = float, help = "image directory")
    parser.add_argument("--batch_size", type = int, help = "batch size")
    parser.add_argument("--result_dir", type = str, help = "image directory")
    parser.add_argument("--device", type = str, help = "cuda:0")
    parser.add_argument("--resolution", type = str, default = "base", help = "low or mid or high") # It make model trained with 128 size.
    return parser.parse_args()

if __name__ == "__main__":

    # get argparser
    args = args()
    face_dir = args.face_dir
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    result_dir = args.result_dir
    device = args.device
    resolution = args.resolution

    # DataLoader
    train_dl = make_dataloader(face_dir, batch_size = batch_size, resolution = resolution) 

    # Generator and Discriminator
    generator = Generator(resolution)
    discriminator = Discriminator(resolution)

    # Set params for trainer
    params = { "Generator" : generator,
               "Discriminator" : discriminator,
               "num_epochs" : num_epochs,
               "loss_function" :  loss_function,
               "train_dl" : train_dl,
               "learning_rate" : lr,
               "result_dir" : result_dir,
               "GPU" : device}

    # Train 😆
    Generator, loss_history, prob_history, fake_image_list = trainer(params)

    # Save Parameters of Generator
    torch.save(Generator.state_dict(), result_dir+'/trained_generators.pt')


    # Plot after training
    plt.title("loss")
    plt.plot(range(1,len(prob_history['D_real'],)+1),loss_history['Generator_loss'], label = "G")
    plt.plot(range(1,len(prob_history['D_real'],)+1),loss_history['Discriminator_loss'], label = "D")
    plt.xlabel("iterations (× 100)")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(result_dir + "/loss.png")

    plt.clf()

    plt.title("prob")
    plt.plot(range(1,len(prob_history['D_real'],)+1), prob_history['D_real'], label = "D(x)")
    plt.plot(range(1,len(prob_history['D_real'],)+1),prob_history['D_G_fake1'], label = "D(G(z))_1")
    plt.plot(range(1,len(prob_history['D_real'],)+1),prob_history['D_G_fake2'], label = "D(G(z))_2")
    plt.xlabel("iterations (× 100)")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(result_dir + "/prob.png")


    for n, iters in enumerate(fake_image_list):
        for m, image in enumerate(iters):   # 16 images
            image = cv2.cvtColor(np.array(np.transpose(image, (1,2,0))), cv2.COLOR_BGR2RGB )
            image = (image>0)*image
            image = cv2.resize(image, dsize=(500, 500), interpolation = cv2.INTER_LINEAR)
            plt.imsave(f"/home/mskang/hyeokjong/GAN/result/images/{m}_{n}_.png", image, dpi=300)
    print("Let's make GIF 😎")
    print("Doing so u can get fantastic sequence of images u made")


    # python train.py --face_dir /home/mskang/hyeokjong/GAN/face/images/img_align_celeba --num_epochs 10 --lr 0.0002 --batch_size 128 --result_dir /home/mskang/hyeokjong/GAN/result --device cuda:1
