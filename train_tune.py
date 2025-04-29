import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('/data/coding/runs/detect/train11/weights/best.pt')      # 需要修改
    model.load('yolov8n.pt') # loading pretrain weights
    model.tune(data=r'/data/coding/ultralytics/data.yaml',
                epochs=50,
                single_cls=False,      # 是否是单类别检测
                batch=16,
                close_mosaic=10,
                device='0',
                optimizer='SGD', 
                project='runs/train',
                iterations=100,
                name='attention_exp',
                )