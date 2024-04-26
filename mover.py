import os
import shutil
import random

def move_images(s, d, sf, n):
    sp = os.path.join(s, sf)
    dp = os.path.join(d, sf)

    os.makedirs(dp, exist_ok=True)

    images = os.listdir(sp)
    random.shuffle(images)
    selected_images = images[:n]

    for image in selected_images:
        sip = os.path.join(sp, image)
        dip = os.path.join(dp, image)
        shutil.move(sip, dip)
        print(f"Moved {image} to {d}/{sf}")


s = "./cell_images/cell_images"
d = "./cell_images/test"
sub_folders = ["Parasitized", "Uninfected"]

n = int(input("How many images do you want to move? "))

for sf in sub_folders:
    move_images(s, d, sf, n)


