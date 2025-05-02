from ultralytics import YOLO
# 加载训练好的模型，改为自己的路径
model = YOLO('/Volumes/Game/ultralytics 8.3.114/runs/detect/attention/weights/best.pt')  #修改为训练好的路径
source = '/Volumes/Game/data/train/JPEGImages/0-lose.jpg' #修改为自己的图片路径及文件名
# 运行推理，并附加参数
model.predict(source, save=True)