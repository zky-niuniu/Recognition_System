import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
	#model = YOLO('/data/coding/ultralytics-8.3.91/ultralytics/cfg/models/v8/yolov8.yaml')   # 修改yaml
	model=YOLO('/data/coding/yolov8m.pt')  #加载预训练权重
	results = model.train(data='./data.yaml',   #数据集yaml文件
	            imgsz=640,
	            epochs=200, #训练轮数测试10,20,40,80,100
	            batch=16,
	            workers=0,  
	            device=0,   #没显卡则将0修改为'cpu'
	            optimizer='SGD',
				close_mosaic=0,
                amp = False,
				plots=True,
	            cache=True,   #服务器可设置为True，训练速度变快
	)