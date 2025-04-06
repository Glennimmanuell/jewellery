import streamlit as st
import cv2
import tempfile
import numpy as np
import os
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import textwrap 

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

sift_data_folder = "sift_dataset"
os.makedirs(sift_data_folder, exist_ok=True)

st.set_page_config(page_title="AR Jewellery", layout="centered")
st.title("💍 Augmented Reality Jewellery Project")

mode = st.sidebar.selectbox("Pilih Mode:", ["Developer", "User"])
temp_dir = tempfile.TemporaryDirectory()

# Ganti ORB ke SIFT
sift = cv2.SIFT_create()

# ---------------- DEVELOPER MODE ----------------
if mode == "Developer":
    st.subheader("🛠️ Developer Mode")
    hidden_message = st.text_area("Masukkan hidden message untuk objek:")

    # ---------------- Kamera Input ----------------
    st.markdown("Ambil gambar dari kamera (bisa lebih dari satu kali):")
    if "captured_images" not in st.session_state:
        st.session_state.captured_images = []

    img_file = st.camera_input("Ambil gambar dari kamera")
    if img_file:
        st.session_state.captured_images.append(img_file)

    if st.session_state.captured_images:
        st.write(f"{len(st.session_state.captured_images)} gambar ditambahkan dari kamera.")
        if st.button("Reset Gambar Kamera"):
            st.session_state.captured_images = []

    # ---------------- File Upload Input ----------------
    st.markdown("---")
    st.markdown("Unggah beberapa gambar dari file (opsional):")
    uploaded_files = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # ---------------- Proses dan Simpan ----------------
    if (st.session_state.captured_images or uploaded_files) and hidden_message:
        process_button = st.button("🔍 Proses & Simpan")

        if process_button:
            all_descriptors = []

            def extract_descriptors(filelist):
                for img in filelist:
                    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, descriptors = sift.detectAndCompute(gray, None)
                    if descriptors is not None:
                        all_descriptors.append(descriptors)

            extract_descriptors(st.session_state.captured_images)
            extract_descriptors(uploaded_files)

            if all_descriptors:
                data = {
                    "message": hidden_message,
                    "descriptors_list": all_descriptors
                }
                filename = os.path.join(sift_data_folder, f"{hidden_message[:10]}_sift.pkl")
                with open(filename, "wb") as f:
                    pickle.dump(data, f)
                st.success(f"{len(all_descriptors)} gambar berhasil disimpan sebagai: {filename}")
                st.session_state.captured_images = []
            else:
                st.warning("Tidak ada fitur yang berhasil diekstrak dari gambar.")

# ---------------- USER MODE ----------------
elif mode == "User":
    st.subheader("📷 User Mode - Object Recognition & QR Code")

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.sift_data = []
            for file in os.listdir(sift_data_folder):
                if file.endswith(".pkl"):
                    with open(os.path.join(sift_data_folder, file), "rb") as f:
                        self.sift_data.append(pickle.load(f))
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            display_img = img.copy()

            # ----------- QR CODE Detection -----------
            qr_detector = cv2.QRCodeDetector()
            data, bbox, _ = qr_detector.detectAndDecode(img)
            if bbox is not None and data:
                pts = np.int32(bbox).reshape(-1, 2)
                for i in range(len(pts)):
                    cv2.line(display_img, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)

                top_center = np.mean([pts[0], pts[1]], axis=0).astype(int)
                x, y = top_center
                y -= 30
                overlay = display_img.copy()
                text = f"🔐 {data}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_w, text_h = text_size
                box_coords = ((x - text_w // 2 - 10, y - text_h - 10), (x + text_w // 2 + 10, y + 10))
                cv2.rectangle(overlay, box_coords[0], box_coords[1], (0, 0, 0), -1)
                display_img = cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0)
                cv2.putText(display_img, text, (x - text_w // 2, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

            # ----------- SIFT Object Detection -----------
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp_frame, des_frame = sift.detectAndCompute(gray, None)

            #display_img = cv2.drawKeypoints(display_img, kp_frame, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            best_score = float("inf")
            best_message = None

            if des_frame is not None:
                for entry in self.sift_data:
                    all_desc_refs = np.vstack(entry["descriptors_list"])
                    matches = self.matcher.match(des_frame, all_desc_refs)
                    score = sum([m.distance for m in matches]) / len(matches) if matches else float("inf")

                    if score < best_score and len(matches) > 30:  # kasih minimal match juga
                        best_score = score
                        best_message = entry["message"]

                if best_message and kp_frame:
                    # Hitung posisi rata-rata dari keypoints
                    pts = np.array([kp.pt for kp in kp_frame], dtype=np.float32)
                    center = tuple(np.mean(pts, axis=0).astype(int))

                    # Bungkus teks agar tidak melebihi batas layar (misal: max 40 karakter per baris)
                    max_chars_per_line = 40
                    wrapped_lines = textwrap.wrap(best_message, width=max_chars_per_line)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    line_height = 30

                    # Hitung ukuran total box
                    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in wrapped_lines]
                    max_line_width = max([w for (w, h) in text_sizes])
                    total_height = line_height * len(wrapped_lines)

                    # Posisi awal box dan teks
                    x, y = center
                    y -= total_height + 20  # Geser sedikit ke atas

                    # Kotak transparan
                    box_x1 = x - max_line_width // 2 - 10
                    box_y1 = y - 10
                    box_x2 = x + max_line_width // 2 + 10
                    box_y2 = y + total_height + 10

                    overlay = display_img.copy()
                    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
                    display_img = cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0)

                    # Tampilkan setiap baris
                    for i, line in enumerate(wrapped_lines):
                        line_y = y + i * line_height + 5
                        line_x = x - cv2.getTextSize(line, font, font_scale, thickness)[0][0] // 2

                        # Outline + teks utama
                        cv2.putText(display_img, line, (line_x, line_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                        cv2.putText(display_img, line, (line_x, line_y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(display_img, format="bgr24")

    webrtc_streamer(
        key="realtime-ar",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False}
    )
