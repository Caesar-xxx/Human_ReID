# 将多个视频搜索结果合并成一个视频
# step:
# 1. 获取query特征
# 2. 加载多个视频的gallery特征，文件名列表
# 3. 分别在多个视频中搜索query
# 4. 将搜索结果合并成一个视频

import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import tqdm
import torch
import pickle
# 使用faiss向量库
import faiss
from PIL import Image
# 提取特征
from model.REID import getImgFeat
# args
import argparse


class GridDemo:
    def __init__(self, query_path, threshold=0.5) :
        # 获取query特征
        self.query_feat = getImgFeat(query_path).numpy()
        self.query_img_data = cv2.imread(query_path)
        # 获取多个视频的gallery特征、文件名列表
        indexs_list = []
        imgs_list = []
        # 只搜索4个视频
        self.query_video_num = 6
        for i in tqdm.tqdm(range(1, 1+self.query_video_num), desc="读取多个视频的gallery特征、文件名列表..."):
            imgs, index = self.loadVideoFeatAndFilenames(i)
            indexs_list.append(index)
            imgs_list.append(imgs)
        # 搜索阈值
        self.threshold = threshold
        # 搜索结果
        self.results_img_list = []
        self.results_dist_list = []
        # 搜索
        for i in tqdm.tqdm(range(self.query_video_num),desc="搜索多个视频中的query..."):
            img_list, D = self.queryVideo(indexs_list[i], imgs_list[i])
            self.results_img_list.append(img_list)
            self.results_dist_list.append(D)

        # 视频cap对象列表
        self.cap_list = []
        for i in tqdm.tqdm(range(1, 1+self.query_video_num), desc="读取视频cap对象..."):
            cap = cv2.VideoCapture('./videos/cam{}.mp4'.format(i))
            self.cap_list.append(cap)
        # 缩放比例
        self.scale = 3
      
    # 在单个视频中搜索query，返回根据frame_id排序的图片列表、置信度（距离）
    def queryVideo(self, index, imgs):
        # 搜索
        lims, D, I = index.range_search(self.query_feat, self.threshold) 
        # 根据索引I找到对应的图片
        # img_list: key为图片索引，value为图片路径
        img_list = {}
        for idx, val in enumerate(I):
            img_list[idx] = imgs[val]

        # 文件根据frame_id排序
        img_list = sorted(img_list.items(), key=lambda x: int(x[1].split("_")[1]))

        # 根据img_list的key，重新对D和I进行排序
        keys = []
        for i in img_list:
            keys.append(i[0])

        # 根据key重新排序，使用的是numpy的fancy indexing，即D[[1,2,3]]，表示取D中索引为1,2,3的元素
        D_new = D[keys]
        # I_new = I[keys]

        return img_list, D_new
        

        
    # 获取单个视频的文件名列表、特征
    def loadVideoFeatAndFilenames(self, video_index):
        # 读取文件名列表
        with open('./weights/cam{}_human_features.pickle'.format(video_index), 'rb') as f:
            data = pickle.load(f)
            imgs = data['imgs']
        # 读取faiss索引
        index = faiss.read_index('./weights/cam{}.index'.format(video_index))

        return imgs, index
    
    # 获取缩放后画面的ltrb
    def getResizedLtrb(self, l, t, r, b):
        # 缩放后的ltrb
        l_new = int(l / self.scale)
        t_new = int(t / self.scale)
        r_new = int(l_new + (r - l) / self.scale)
        b_new = int(t_new + (b - t) / self.scale)
        return l_new, t_new, r_new, b_new
    # 获取视频某一帧
    # 参数：video_index: 视频索引、offset_id: 偏移量
    def getVideoFrame(self, video_index, offset_id):
        # 检查偏移量是否越界
        if offset_id >= len(self.results_img_list[video_index]):
            print("camera {} offset_id {} out of range".format(video_index+1, offset_id))
            return None, None, None
        # 获取offset_id对应的frame、图片索引
        id, frame = self.results_img_list[video_index][offset_id]
        frame_id = int(frame.split("_")[1]) # 原始画面的帧号
        # ltrb
        l = int(frame.split("_")[2])
        t = int(frame.split("_")[3])
        r = int(frame.split("_")[4])
        b = int(frame.split("_")[5].split(".")[0])
        # 缩放后的ltrb
        l, t, r, b = self.getResizedLtrb(l, t, r, b)
        # jump to frame
        self.cap_list[video_index].set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        # 读取
        ret, video_frame = self.cap_list[video_index].read()
        # 缩放为原来的1/self.scale
        video_frame = cv2.resize(video_frame, (video_frame.shape[1]//self.scale, video_frame.shape[0]//self.scale))
        # draw rectangle
        cv2.rectangle(video_frame, (l, t), (r, b), (0, 255, 0), 2)
        # draw similarity, x100再保留两位小数
        similarity = round(self.results_dist_list[video_index][id]*100, 2)
        cv2.putText(video_frame, str(similarity) + "%", ( l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # draw camera id
        cv2.putText(video_frame, "camera: " + str(video_index+1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # draw frame id
        cv2.putText(video_frame, "frame: " + str(frame_id), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # img_frame
        img_frame = cv2.imread(frame)
        return video_frame , img_frame, similarity

       


    # 将搜索结果合并成一个视频
    def run(self):
        # 读取视频，跳转到指定帧
        offset_id = 0
        # resize
        resize_w = 720
        resize_h = 400
        # 当前黑屏的窗口，状态为True
        black_window_num = {}

        # writer
        writer = cv2.VideoWriter(
            "./output/grid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (resize_w*2, 150+resize_h*3)
        )
        
        while True:
            # 创建一个宽度为resize_w*2，高度为150的黑色画布
            canvas = np.zeros((150, resize_w * 2, 3), dtype=np.uint8)
            # 缩放query图片为高度120, 宽度按比例缩放
            query_ratio = self.query_img_data.shape[1] / self.query_img_data.shape[0]
            query_img = cv2.resize(self.query_img_data,  (int(100*query_ratio), 100))
            # 将query图片贴到画布左侧中心
            canvas[15:15+query_img.shape[0], 100:100+query_img.shape[1]] = query_img
            # 下方加query图片的文字说明
            cv2.putText(canvas, "query", (90, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
            # 读取6个视频的frame
            frame_list = []
            for i in range(self.query_video_num):
                video_frame , img_frame, similarity = self.getVideoFrame(i, offset_id)
                if video_frame is None:
                    # 用黑色填充 720 x 480
                    video_frame = np.zeros((resize_h, resize_w, 3), dtype=np.uint8)
                    # 在黑色背景正中间绘制文字：camera: i+1 end
                    cv2.putText(video_frame, "camera: " + str(i+1) + " end", (int(resize_w/2)-100, int(resize_h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    black_window_num[i] = True
                    # 如果6个窗口都是黑的，则退出
                    if len(black_window_num) == self.query_video_num:
                        return
                    
                else:
                    # 统一缩放为720x480
                    video_frame = cv2.resize(video_frame, (resize_w, resize_h))
                    # 将img_frame按照camera_id贴到画布右侧
                    # 缩放为高度120, 宽度按比例缩放
                    img_ratio = img_frame.shape[1] / img_frame.shape[0]
                    img_frame = cv2.resize(img_frame,  (int(100*img_ratio), 100))
                    # 将img_frame贴到画布右侧中心
                    l = 300 + 200*i
                    t = 15
                    r = l + img_frame.shape[1]
                    b = t + img_frame.shape[0]
                    canvas[t:b, l:r] = img_frame
                    # 下方显示相似度
                    cv2.putText(canvas, str(similarity) + "%", (l-10, b+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                frame_list.append(video_frame)
            # 拼接为3row x 2col
            frame1 = np.hstack((frame_list[0], frame_list[1]))
            frame2 = np.hstack((frame_list[2], frame_list[3]))
            frame3 = np.hstack((frame_list[4], frame_list[5]))
            frame_v = np.vstack((canvas, frame1, frame2, frame3))
            
            # 写入
            writer.write(frame_v)
            # 显示，缩放
            cv2.imshow("frame", frame_v)

            offset_id += 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
           

       



# 运行
# query
test_img = './images/cam1_177_1346_367_1584_1197.jpg'
# test_img = './images/cam1_227_1284_335_1573_1127.jpg'
# test_img = './images/cam1_9302_848_180_1009_739.jpg'
parser = argparse.ArgumentParser(description='grid_video')
parser.add_argument('--query_path', type=str, default=test_img, help='query image path')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
args = parser.parse_args()
grid_demo = GridDemo(query_path = args.query_path , threshold=args.threshold)
grid_demo.run()

