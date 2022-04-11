from PIL import Image
import os
import argparse
from tqdm import tqdm


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type = str, help = "Generated image directory")
    parser.add_argument("--GIF_dir", type = str, help = "GIF directory to saved")
    return parser.parse_args()

def make_GIF(GIF_dir ,images_dir):
    list_ = os.listdir(images_dir)

    # for resized images
    for n in tqdm(range(16)):
        frames = []
        for i in list_:
            if (i[0:2] == (str(n) + "_" )) or (i[0:2] == str(n)):
                frames.append(Image.open(images_dir+"/"+i))
        GIF = frames[0]
        GIF.save(f"{GIF_dir}/{n}th.gif", format = "GIF", append_images = frames,
        save_all = True, duration = 100, loop = 0, dpi=(500, 500))

    # for raw images
    for n in tqdm(range(16)):
        frames = []
        for i in list_:
            if ( ((i[4:6] == (str(n) + "_" )) or (i[4:6] == str(n))) and i[0:3]=="raw"):
                frames.append(Image.open(images_dir+"/"+i))
        GIF = frames[0]
        GIF.save(f"{GIF_dir}/raw_{n}th.gif", format = "GIF", append_images = frames,
        save_all = True, duration = 100, loop = 0, dpi=(500, 500))




if __name__ == "__main__":

    # get argparser
    args = args()
    images_dir = args.images_dir
    GIF_dir = args.GIF_dir

    make_GIF(GIF_dir ,images_dir)

# python make_GIF.py --images_dir /home/mskang/hyeokjong/GAN/result/images --GIF_dir /home/mskang/hyeokjong/GAN/result/GIF
