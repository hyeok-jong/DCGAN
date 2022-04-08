from PIL import Image
import os
import argparse


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type = str, help = "Generated image directory")
    parser.add_argument("--GIF_dir", type = str, help = "GIF directory to saved")
    return parser.parse_args()

def make_GIF(GIF_dir ,images_dir):
    list_ = os.listdir(images_dir)

    for n in range(16):
        frames = []

        for i in list_:

            if (i[0] == str(n)) or(i[0:2] == str(n)):
                frames.append(Image.open(images_dir+"/"+i))
        GIF = frames[0]
        GIF.save(f"{GIF_dir}/{n}th.gif", format = "GIF", append_images = frames,
        save_all = True, duration = 200, loop = 0, dpi=(300, 300))



if __name__ == "__main__":

    # get argparser
    args = args()
    images_dir = args.images_dir
    GIF_dir = args.GIF_dir

    make_GIF(GIF_dir ,images_dir)

# python make_GIF.py --images_dir /home/mskang/hyeokjong/GAN/result/images --GIF_dir /home/mskang/hyeokjong/GAN/result/GIF
