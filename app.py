import streamlit as st
from fer import FER
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Deteksi Emosi Wajah", layout="centered")
st.title("😊 Aplikasi Deteksi Ekspresi Wajah")

# ⬇️ Tambahkan bagian ini setelah judul
st.markdown("""
### Tentang Aplikasi
Aplikasi ini menggunakan model **Convolutional Neural Network (CNN)** melalui pustaka `FER` (Facial Expression Recognition) untuk mendeteksi ekspresi emosi dari wajah seseorang pada gambar.

### Dibuat oleh:
1. **Alyaa Nidya Shadrina Anwar**  
2. **Puspa Damai Kukuh Hati**  
3. **Maziyah Mufidah Wahyudiono**

**Mahasiswa Program Studi Statistika Bisnis**  
Fakultas Vokasi, Institut Teknologi Sepuluh Nopember

### Cara Menggunakan:
1. Unggah foto wajah yang ingin dianalisis.
2. Tunggu hasil deteksi emosi ditampilkan.
3. Gambar akan ditandai dengan ekspresi dominan.

---
""")

# Lanjutkan bagian ini seperti biasa
uploaded_file = st.file_uploader("📤 Unggah foto wajah", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    detector = FER(mtcnn=True)
    result = detector.detect_emotions(img_bgr)

    if result:
        st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
        for face in result:
            box = face["box"]
            emotions = face["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            st.success(f"Ekspresi terdeteksi: **{top_emotion}** (confidence: {emotions[top_emotion]:.2f})")

            (x, y, w, h) = box
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_array, top_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        st.image(img_array, caption="Deteksi Emosi", use_column_width=True)
    else:
        st.warning("⚠️ Tidak ada wajah terdeteksi pada gambar.")
