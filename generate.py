from generator import Generator
import torch
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type = str, help = "trained_generators.pt")
    parser.add_argument("--save_dir", type = str, help = "save")
    parser.add_argument("--GPU", type = str, help = "cuda:0")
    return parser.parse_args()


if  __name__ == "__main__":

    file = arg().pt_dir
    device = arg().GPU
    dir = arg().save_dir

    Generator = Generator().to(device)
    Generator.load_state_dict(torch.load(file)) 

    noise_fixed = torch.randn(16, 100, 1, 1, device = device)

    ''' 🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓
    noise_fixed = torch.randn(16, 100, 7, 7, device = device)
    🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓 '''

    with torch.no_grad():
        fake_image_generated = Generator(noise_fixed).detach().cpu()

    for m, image in enumerate(fake_image_generated):
        image = cv2.cvtColor(np.array(np.transpose(image, (1,2,0))), cv2.COLOR_BGR2RGB )
        image = (image>0)*image
        plt.imsave(f"{dir}/_{m}_after.png", image, dpi=300)
