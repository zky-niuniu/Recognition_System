import streamlit as st
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import datetime
import time

# 配置区域 ================================================
SAVE_DIR = "/Volumes/Game/Saved_results" #设置成你的保存地址
os.makedirs(SAVE_DIR, exist_ok=True)

# 初始化Session State
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'last_save_time' not in st.session_state:
    st.session_state.last_save_time = 0

# 侧边栏配置 =============================================
st.sidebar.title("Settings")
#权重文件地址
MODEL_PATH = st.sidebar.text_input("Model Path", "runs/detect/attention/weights/best.pt")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.05)
auto_save = st.sidebar.checkbox("Enable Auto-Save", True)
#限制两次保存之间的最小时间间隔，避免存储耗尽与频繁IO操作导致的性能下降
#camera模块，用不到摄像头实际上没有用处
#ave_cooldown = st.sidebar.number_input("Save Cooldown (seconds)", 1, 60, 5)

# 加载模型
try:
    model = YOLO(MODEL_PATH)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model loaded failed: {e}")
    st.stop()

# 图像处理函数 ===========================================
def process_image(image, conf_threshold):
    results = model(image)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            conf = float(box.conf[0]) if hasattr(box.conf, '__iter__') else float(box.conf)
            
            if conf >= conf_threshold:
                r = box.xyxy[0].astype(int)
                cls = int(box.cls[0]) if hasattr(box.cls, '__iter__') else int(box.cls)
                label = f'{model.names[cls]} {conf:.2f}'
                
                cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)
                cv2.putText(image, label, (r[0], r[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image, results

# 帧处理函数 ===========================================
def process_frame(frame, conf_threshold):
    results = model(frame)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            conf = float(box.conf[0]) if hasattr(box.conf, '__iter__') else float(box.conf)
            if conf >= conf_threshold:
                r = box.xyxy[0].astype(int)
                cls = int(box.cls[0]) if hasattr(box.cls, '__iter__') else int(box.cls)
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (r[0], r[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame, len(boxes) > 0

# 视频处理主函数 =========================================
def process_video(video_path, conf_threshold):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建临时输出文件
    output_path = os.path.join(SAVE_DIR, f"processed_{int(time.time())}.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # 创建显示容器
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    detection_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 处理帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, has_detection = process_frame(frame_rgb, conf_threshold)
        
        # 写入输出视频
        out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        
        # 更新显示（每5帧更新一次以提高性能）
        if processed_frames % 5 == 0:
            frame_placeholder.image(processed_frame, channels="RGB")
        
        # 更新计数
        if has_detection:
            detection_count += 1
        processed_frames += 1
        
        # 更新进度
        progress = min(processed_frames / total_frames, 1.0)
        progress_bar.progress(progress)
        status_text.text(
            f"Processed: {processed_frames}/{total_frames} frames | "
            f"Detections: {detection_count} | "
            f"FPS: {fps:.1f}"
        )
    
    # 释放资源
    cap.release()
    out.release()
    return output_path, detection_count
# 主界面 ================================================
st.title("Manhole Cover Recognition System")
option = st.radio("Input Type", ("Image", "Video"), horizontal=True)

# 图像处理分支
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            try:
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption="Original Image", use_container_width=True)
            except Exception as e:
                st.error(f"image load error: {e}")
        
        with col2:
            try:
                image_np = np.array(original_image)
                if len(image_np.shape) == 2:  # 灰度图
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                elif image_np.shape[2] == 4:   # RGBA图
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                
                processed_image, results = process_image(image_np, conf_threshold)
                st.image(processed_image, caption="Processed Image", use_container_width=True)
                
                if st.button('Save Result'):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(SAVE_DIR, f"result_{timestamp}.jpg")
                    Image.fromarray(processed_image).save(save_path)
                    st.toast(f"Saved to {save_path}", icon="✅")
                    
                    # 显示检测结果
                    if results and len(results[0].boxes) > 0:
                        st.json({
                            "detections": len(results[0].boxes),
                            "classes": [model.names[int(box.cls)] for box in results[0].boxes],
                            "confidences": [float(box.conf) for box in results[0].boxes]
                        })
            except Exception as e:
                st.error(f"image error: {e}")

# 视频处理分支
elif option == "Video":
    uploaded_file = st.file_uploader("upload video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # 保存上传的视频到临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        
        # 显示原始视频
        st.video(uploaded_file)
        
        if st.button("Start to progress video"):
            with st.spinner("progressing..."):
                start_time = time.time()
                processed_path, detections = process_video(tfile.name, conf_threshold)
                elapsed = time.time() - start_time
                
                st.success(f"progress finish! consuming: {elapsed:.2f}second | Target detected: {detections}times")
                st.video(processed_path)
        
        # 清理临时文件
        os.unlink(tfile.name)

# 摄像头处理分支，可以用于后续功能扩展：
#option = st.radio("Input Type", ("Image", "Video"), horizontal=True)加上"camera"即可
#elif option == "Camera":
#    st.warning("Camera may take a few seconds to initialize")
    
#    if st.button("Start Camera", disabled=st.session_state.camera_active):
#        st.session_state.camera_active = True
    
#    if st.button("Stop Camera", disabled=not st.session_state.camera_active):
#        st.session_state.camera_active = False
#        st.experimental_rerun()
    
#    if st.session_state.camera_active:
#        FRAME_WINDOW = st.empty()
#        camera = cv2.VideoCapture(0)
        
#        try:
#            while st.session_state.camera_active:
#                ret, frame = camera.read()
#                if not ret:
#                    st.error("Failed to capture frame from camera")
#                    st.session_state.camera_active = False
#                    break
                
#                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                processed_frame, results = process_image(frame_rgb, conf_threshold)
#                FRAME_WINDOW.image(processed_frame)
                
                #自动保存
#                if auto_save and results and len(results[0].boxes) > 0:
#                    current_time = time.time()
#                    if current_time - st.session_state.last_save_time > save_cooldown:
#                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#                        save_path = os.path.join(SAVE_DIR, f"camera_{timestamp}.jpg")
#                        cv2.imwrite(save_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
#                        st.session_state.last_save_time = current_time
#                        st.sidebar.success(f"Detection saved: {save_path}")
        
#        except Exception as e:
#            st.error(f"Camera error: {e}")
#            st.session_state.camera_active = False
#        finally:
#            camera.release()
#            if st.session_state.camera_active:
#                st.session_state.camera_active = False
#                st.experimental_rerun()

# 添加页脚
st.markdown("---")
st.caption("Manhole Cover Detection System | yzk")