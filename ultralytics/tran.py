from ultralytics import YOLO
 
# Load a model
model = YOLO(r"/Volumes/Game/ultralytics-8.3.91/runs/detect/train5/weights/best.pt")  # load a custom trained model
 
 
# Export the model to NCNN with arguments
success = model.export(format="onnx")  # creates '/yolo11n_ncnn_model'
