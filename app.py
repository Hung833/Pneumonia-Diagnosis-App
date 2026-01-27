        
import streamlit as st, tensorflow as tf, numpy as np

from PIL import Image


#1.Setup title page
st.set_page_config(page_title="AI Bác Sĩ", page_icon="🩺")
st.title("🩺 Hệ Thống Chuẩn Đoán Viêm Phổi Bằng AI")
st.write("Vui lòng upload ảnh X-quang phổi để hệ thống phân tích")

#2.Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('pneumonia_model_v1.keras')
    return model

with st.spinner("Đang khởi động bộ não AI..."):
    model = load_model()

#3.Upload image file
uploaded_file = st.file_uploader("Chọn ảnh X-quang (đuôi jpg, png, jpeg)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh X-quang đã tải lên", use_container_width=True)
    
    
    #4.Predict button
    if  st.button("🔍 Phân tích ngay"):
        with st.spinner("AI đang soi phim..."):

            image_gray = img.convert('L')
            image_resized = image_gray.resize((224, 224))
            img_array = np.array(image_resized)
            img_array = img_array / 255.0
            img_array = img_array.reshape(-1, 224, 224, 1)

            #5.Predict
            prediction = model.predict(img_array)

            score_normal = prediction[0][0] * 100     
            score_pneumonia = prediction[0][1] * 100

            #6.Display results
            st.write("---")
            st.subheader("Kết quả chuẩn đoán:")

            if score_pneumonia > 50:
                st.error(f"⚠️ CẢNH BÁO: PHÁT HIỆN DẤU HIỆU BỊ VIÊM PHỔI")
                st.write(f"Độ tin cậy: **{score_pneumonia:.2f}%**")
                st.progress(int(score_pneumonia))

            else:
                st.success(f"✅ KẾT QUẢ: PHỔI BÌNH THƯỜNG")
                st.write(f"Độ tin cậy: **{score_normal:.2f}%**")
                st.progress(int(score_normal))

            st.info("Lưu ý: Kết quả này chỉ mang tính tham khảo kỹ thuât, không thể thay thế chuẩn đoán của Bác sĩ.")