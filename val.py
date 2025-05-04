from ultralytics import YOLO
import matplotlib
matplotlib.use( "TkAgg")
if __name__ == '__main__':
    #加载训练好的模型
    model = YOLO('/Volumes/Game/ultralytics 8.3.114/runs/detect/main_net/weights/best.pt')
    # 对验证集进行评估
    metrics = model.val(data = '/Volumes/Game/ultralytics 8.3.114/data.yaml')