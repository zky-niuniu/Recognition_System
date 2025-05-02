from ultralytics import YOLO
if __name__=="__main__":
    # Load a model
    pth_path=r"/Volumes/Game/ultralytics 8.3.114/runs/detect/attention/weights/best.pt"
    #models need to Validate:
        #/Volumes/Game/ultralytics 8.3.114/runs/detect/attention/weights/best.pt
        #/Volumes/Game/ultralytics 8.3.114/runs/detect/attention_2/weights/best.pt
        #/Volumes/Game/ultralytics 8.3.114/runs/detect/main_net/weights/best.pt
        #/Volumes/Game/ultralytics 8.3.114/runs/detect/main_net_2/weights/best.pt
        #/Volumes/Game/ultralytics 8.3.114/runs/detect/original/weights/best.pt
        #/Volumes/Game/ultralytics 8.3.114/runs/detect/eucalyptus_advanced/weights/best.pt
    model = YOLO(pth_path)  # load a custom model
    # Validate the model
    metrics = model.val(data="/Volumes/Game/ultralytics 8.3.114/data.yaml")
    print(metrics.box.map)    # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)   # a list contains map50-95 of each category