import streamlit as st
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import datetime
import time
import io 

os.environ["TORCH_FORCE_WEIGHTS_ONLY"] = "0"

# 配置区域 ================================================
SAVE_DIR = os.path.join(tempfile.gettempdir(), "recognition_results")
os.makedirs(SAVE_DIR, exist_ok=True)

# 初始化Session State
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'last_save_time' not in st.session_state:
    st.session_state.last_save_time = 0

# 侧边栏配置 =============================================
st.sidebar.title("Settings")
#权重文件地址
MODEL_PATH = st.sidebar.text_input("Model Path", "runs/detect/original/weights/best.pt")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.05)
auto_save = st.sidebar.checkbox("Enable Auto-Save", True)

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
                st.image(original_image, caption="Original Image", use_container_width=True)  # 修改这里
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
                st.image(processed_image, caption="Processed Image", use_container_width=True)  # 修改这里
                
                # 将处理后的图像转换为字节流以供下载
                processed_pil = Image.fromarray(processed_image)
                img_byte_arr = io.BytesIO()
                processed_pil.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                if st.download_button(
                    label='Download Result',
                    data=img_byte_arr,
                    file_name=f"processed_{uploaded_file.name}",
                    mime="image/jpeg"
                ):
                    st.toast("Download started!", icon="✅")
                    
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
        # 显示原始视频
        st.video(uploaded_file)
        
        # 保存上传的视频到临时文件
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Start to process video"):
            with st.spinner("Processing..."):
                start_time = time.time()
                try:
                    processed_path, detections = process_video(temp_path, conf_threshold)
                    elapsed = time.time() - start_time
                    
                    st.success(f"Process finished! Time taken: {elapsed:.2f} seconds | Detections: {detections}")
                    
                    # 显示处理后的视频
                    with open(processed_path, "rb") as f:
                        video_bytes = f.read()
                    st.video(video_bytes)
                    
                    # 提供下载链接
                    st.download_button(
                        label="Download Processed Video",
                        data=video_bytes,
                        file_name=f"processed_{uploaded_file.name}",
                        mime="video/mp4"
                    )
                except Exception as e:
                    st.error(f"Video processing error: {e}")
                finally:
                    # 清理临时文件
                    try:
                        os.remove(temp_path)
                        if 'processed_path' in locals():
                            os.remove(processed_path)
                    except Exception as e:
                        st.warning(f"Failed to clean up temporary files: {e}")

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
