# 将视频中的人体全部裁剪出来，保存到images文件夹中
# 命名规则：camIndex_frameIndex_l_t_r_b.jpg

# 导入相关库
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import glob

# 加载模型
model_file = "./person_s/s_100_800/weights/best.pt"
# model_file = "./weights/yolov8s.pt"
model = YOLO(model_file)  
objs_labels = model.names  
print(objs_labels)

def videoCropPersons(video_file):
    cam_id = video_file.split(os.sep)[-1].split('.')[0]
    # 读取视频
    cap = cv2.VideoCapture(video_file)
    # get frame count
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video: {video_file}, frame_count: {frame_count}, width: {width}, height: {height}, fps: {fps}" )

    while True:
        # 读取一帧
        start_time = time.time()
        ret, frame = cap.read()

        if ret:
            # 检测
            result = list(model(frame, stream=True, conf=0.4))[0]  # inference，如果stream=False，返回的是一个列表，如果stream=True，返回的是一个生成器
            boxes = result.boxes  # Boxes object for bbox outputs
            boxes = boxes.cpu().numpy()  # convert to numpy array
            
            # 遍历每个框
            for box in boxes.data:
                l,t,r,b = box[:4].astype(np.int32) # left, top, right, bottom
                conf, id = box[4:] # confidence, class
                # if id == 0: # person, save frame to images folder
                if id == 0:
                    # cv2.rectangle(frame, (l,t), (r,b), (0,0,255), 2)
                    # # cv2.putText(frame, f"{objs_labels[id]} {conf*100:.1f}%", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv2.putText(frame, f"{conf*100:.1f}%", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # save frame
                    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    # print(frame_index)
                    img_name = f"./images/{cam_id}_{frame_index}_{l}_{t}_{r}_{b}.jpg"
                    cv2.imwrite(img_name, frame[t:b, l:r])
                
            # end time
            end_time = time.time()
            # FPS
            fps = 1 / (end_time - start_time)
            # 绘制FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    

            # 显示
            # 缩放0.5倍
            # frame = cv2.resize(frame, (int(width/2), int(height/2)))
            # cv2.imshow("frame", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            print("video end")
            # release
            cap.release()
            # cv2.destroyAllWindows()
            break
# mkdir
if not os.path.exists("./images"):
    os.mkdir("./images")

videos = glob.glob("./videos/*.mp4")
for video_file in videos:
    videoCropPersons(video_file)