from ultralytics import YOLO
model = YOLO('/Volumes/Game/ultralytics 8.3.114/runs/detect/main_net_2/weights/best.pt')
# # success = model.export(format="onnx",imgsz=128,half = True,optimize =True,device="cpu") # wrong
success = model.export(format="onnx",imgsz=640,device="cpu", opset=12,simplify=True)
# # Export the model to NCNN format
model.export(format="ncnn",imgsz=640,device="cpu")  # creates '/yolov8n_ncnn_model'