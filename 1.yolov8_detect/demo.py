
# 导入相关库
from ultralytics import YOLO
import cv2
import numpy as np
import time

# 加载模型
model_file = "./person_s/n_100_800/weights/best.pt"
# model_file = "./weights/yolov8x.pt"
model = YOLO(model_file)  
objs_labels = model.names  
print(objs_labels)


video_file = "./videos/D05_20230707102820.mp4"
# 读取视频
cap = cv2.VideoCapture(video_file)
# get frame count
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# 获取视频的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"width: {width}, height: {height}, fps: {fps}, frame_count: {frame_count}")

while True:
    # 读取一帧
    start_time = time.time()
    ret, frame = cap.read()

    if ret:
        # resize to half
        frame = cv2.resize(frame, (int(width/2), int(height/2)))
        # 检测
        result = list(model(frame, stream=True, conf=0.4))[0]  # inference，如果stream=False，返回的是一个列表，如果stream=True，返回的是一个生成器
        boxes = result.boxes  # Boxes object for bbox outputs
        boxes = boxes.cpu().numpy()  # convert to numpy array
        
        # 参考：https://docs.ultralytics.com/modes/predict/#boxes
        # 遍历每个框
        for box in boxes.data:
            l,t,r,b = box[:4].astype(np.int32) # left, top, right, bottom
            conf, id = box[4:] # confidence, class
            # print(f"l: {l}, t: {t}, r: {r}, b: {b}, conf: {conf}, id: {id}")
            # if id == 0: # person, save frame to images folder
            if id == 0:
                cv2.rectangle(frame, (l,t), (r,b), (0,0,255), 2)
                # cv2.putText(frame, f"{objs_labels[id]} {conf*100:.1f}%", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"{conf*100:.1f}%", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # end time
        end_time = time.time()
        # FPS
        fps = 1 / (end_time - start_time)
        # 绘制FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                

        # 显示
        # 缩放0.5倍
        frame = cv2.resize(frame, (int(width/2), int(height/2)))
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
