
import os
import glob
import matplotlib.pyplot as plt
import tqdm
import torch
import pickle
# 使用faiss向量库
import faiss
# 提取特征
from model.REID import getImgFeat

def extractFeats(cam_index):
        
    # 找出images下所有cam_index开头的图片
    images_list = glob.glob('./images/cam{}*.jpg'.format(cam_index))

    person_feats_1 = []
    person_feats_2 = []
    person_feats_3 = []
    pickle_filename = './weights/cam{}_human_features.pickle'.format(cam_index)
    # check cam1_human_features.pickle exist
    if not os.path.exists(pickle_filename):
        # 读取文件夹下所有图片
        # 分成3段来处理，每段10000张图片
        for img in tqdm.tqdm(images_list[:10000]):
            feat = getImgFeat(img)
            person_feats_1.append(feat)
        for img in tqdm.tqdm(images_list[10000:20000]):
            feat = getImgFeat(img)
            person_feats_2.append(feat)
        for img in tqdm.tqdm(images_list[20000:]):
            feat = getImgFeat(img)
            person_feats_3.append(feat)    

        # 拼接三段
        person_feats = person_feats_1 + person_feats_2 + person_feats_3
        human_feats = torch.cat(person_feats, 0) # 将所有特征拼接起来
        print(human_feats.shape)

        # 写入文件
        with open(pickle_filename, "wb") as f:
            pickle.dump({'human_feats': human_feats, 'imgs': images_list}, f)

        # 将特征转换为numpy格式
        human_feats_np = human_feats.numpy()
        print(human_feats_np.shape)
        # 创建索引
        index = faiss.IndexFlatIP(human_feats_np.shape[1])
        # 添加数据
        index.add(human_feats_np)
        print(index.ntotal)
        # save index
        faiss.write_index(index, './weights/cam{}.index'.format(cam_index))



# 2-7的视频文件
for cam_index in range(1, 7):
    # print('cam_index: ', cam_index)
    extractFeats(cam_index)


