from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/Volumes/Game/ultralytics-8.3.91/yolov8n.pt")

# Export the model to NCNN format
model.export(format="ncnn",)  # creates '/yolo11n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("/Volumes/Game/ultralytics-8.3.91/yolo11n_ncnn_model")

# Run inference
results = ncnn_model("https://ultralytics.com/images/bus.jpg")