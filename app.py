import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.cm as cm
import os
import imutils

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Thống Chẩn Đoán Đa Model", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; color: #4CAF50; }
    .error-font { font-size:20px !important; font-weight: bold; color: #FF5252; }
    .title-text { text-align: center; font-size: 40px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CÁC HÀM XỬ LÝ ẢNH ---

# A. Xử lý cho Model PRO (Cắt sọ + RGB)
def ham_xu_ly_cho_PRO(img):
    if img.dtype != 'uint8':
        img = np.array(img, dtype=np.uint8)

    # Cắt xương sọ
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    else:
        new_img = img 

    # Resize giữ tỷ lệ + Padding
    desired_size = 128
    old_size = new_img.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    new_img = cv2.resize(new_img, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0] 
    new_img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img

# B. Xử lý cho Model FINAL (Ảnh Xám + Resize thường)
def ham_xu_ly_cho_FINAL(img_bgr):
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Resize thẳng tay về 128x128
    resized = cv2.resize(gray, (128, 128))
    return resized

# --- 3. HÀM LOAD MODEL ---
@st.cache_resource
def load_model_by_name(model_name):
    if model_name == "PRO":
        path = 'brain_tumor_PRO.h5'
    else:
        path = 'brain_tumor_FINAL.h5'
        
    if not os.path.exists(path):
        return None
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        return None

# --- 4. CÁC HÀM HEATMAP (ĐÃ SỬA LỖI "NEVER CALLED") ---
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if 'conv2d' in layer.name:
            return layer.name
    return None

# 👇 ĐÂY LÀ HÀM QUAN TRỌNG ĐÃ SỬA 👇
def make_gradcam_heatmap_manual(img_tensor, model, last_conv_layer_name):
    # Đảm bảo input là tensor float32
    img_tensor = tf.cast(img_tensor, tf.float32)

    with tf.GradientTape() as tape:
        x = img_tensor
        last_conv_output = None
        
        # Vòng lặp thủ công qua từng lớp để tránh lỗi Graph
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                last_conv_output = x
                tape.watch(last_conv_output)
        
        preds = x
        top_pred_index = tf.argmax(preds[0])
        class_channel = preds[:, top_pred_index]

    # Tính toán Gradient
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Nếu ảnh nền là xám (2 chiều), convert sang RGB để trộn màu
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

# --- 5. GIAO DIỆN CHÍNH ---

st.markdown('<p class="title-text">🧠 AI Chẩn Đoán U Não</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Cấu Hình")
model_option = st.sidebar.radio(
    "Chọn Phiên Bản AI:",
    ("Model PRO (Mới - Cắt Sọ)", "Model FINAL (Cũ - Ảnh Xám)")
)

current_model_name = "PRO" if "PRO" in model_option else "FINAL"
model = load_model_by_name(current_model_name)

if model is None:
    st.sidebar.error(f"❌ Không tìm thấy file cho {current_model_name}!")
else:
    st.sidebar.success(f"✅ Đã load {current_model_name}")

st.write("---")

# --- MAIN ---
if model is not None:
    uploaded_file = st.sidebar.file_uploader("Upload ảnh MRI:", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col1, col2, col3 = st.columns([1, 1, 1])
        final_input_tensor = None
        img_for_display = None
        
        with col1:
            st.info("📷 Ảnh Gốc")
            st.image(img_rgb, width=250)

        # --- LOGIC XỬ LÝ ---
        if current_model_name == "PRO":
            # PRO: Cắt sọ, RGB
            processed_img = ham_xu_ly_cho_PRO(img_rgb)
            img_for_display = processed_img
            img_normalized = processed_img.astype('float32') / 255.0
            final_input_tensor = np.expand_dims(img_normalized, axis=0) # Shape (1, 128, 128, 3)
            
            with col2:
                st.success("✨ PRO: Đã Cắt Sọ")
                st.image(processed_img, width=250, caption="Input sạch sẽ")

        else: 
            # FINAL: Xám, Resize thường
            processed_img = ham_xu_ly_cho_FINAL(img_bgr)
            img_for_display = processed_img
            img_normalized = processed_img.astype('float32') / 255.0
            img_expanded = np.expand_dims(img_normalized, axis=-1) 
            final_input_tensor = np.expand_dims(img_expanded, axis=0) # Shape (1, 128, 128, 1)
            
            with col2:
                st.warning("⚠️ FINAL: Ảnh Xám")
                st.image(processed_img, width=250, caption="Input thô (chỉ resize)")

        analyze = st.button("🚀 CHẨN ĐOÁN NGAY", type="primary")

        if analyze:
            with st.spinner(f'Model {current_model_name} đang chạy...'):
                try:
                    # 1. Dự đoán
                    preds = model.predict(final_input_tensor)
                    score = preds[0][0]
                    
                    # 2. Vẽ Heatmap (Đã sửa lỗi)
                    last_layer = get_last_conv_layer_name(model)
                    if last_layer:
                        heatmap = make_gradcam_heatmap_manual(final_input_tensor, model, last_layer)
                        final_heatmap_img = overlay_heatmap(img_for_display, heatmap)
                        with col3:
                            st.error("🔥 Heatmap Vùng Bệnh")
                            st.image(final_heatmap_img, width=250)
                    else:
                        st.warning("Không tìm thấy layer Conv2D để vẽ Heatmap")
                    
                    # 3. Kết luận
                    st.write("---")
                    st.subheader(f"📊 KẾT QUẢ TỪ {current_model_name}:")
                    
                    # Threshold riêng cho từng model
                    threshold = 0.2 if current_model_name == "PRO" else 0.5
                    
                    if score > threshold:
                        label = "CÓ U (YES)"
                        conf = score * 100
                        st.markdown(f'<p class="error-font">⚠️ PHÁT HIỆN: {label}</p>', unsafe_allow_html=True)
                        st.progress(int(conf))
                        st.write(f"Độ tin cậy: **{conf:.2f}%** (Ngưỡng: {threshold})")
                    else:
                        label = "KHÔNG U (NO)"
                        conf = (1 - score) * 100
                        st.markdown(f'<p class="big-font">✅ KẾT LUẬN: {label}</p>', unsafe_allow_html=True)
                        st.progress(int(conf))
                        st.write(f"Độ tin cậy: **{conf:.2f}%** (Ngưỡng: {threshold})")

                except Exception as e:
                    st.error(f"Lỗi hệ thống: {e}")
                    # In chi tiết lỗi để debug nếu cần
                    import traceback
                    st.text(traceback.format_exc())
    else:
        st.info("👈 Mời upload ảnh để bắt đầu.")