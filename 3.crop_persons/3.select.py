# 在images 中随机挑选1W张图片，复制存入selected_imgs文件夹中
import os
import glob
import random
import shutil
import cv2
import tqdm


def select_imgs(imgs_dir, selected_dir, num=10000):
    if not os.path.exists(selected_dir):
        os.makedirs(selected_dir)
    imgs = glob.glob(os.path.join(imgs_dir, "*.jpg"))
    random.shuffle(imgs)
    for img in tqdm.tqdm(imgs[:num]):
        shutil.copy(img, selected_dir)

# 运行
select_imgs("./images", "./selected_imgs", num=1000)