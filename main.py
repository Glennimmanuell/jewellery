import streamlit as st
import qrcode
import cv2
import tempfile
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="AR Jewellery", layout="centered")
st.title("💍 Augmented Reality Jewellery Project")

mode = st.sidebar.selectbox("Pilih Mode:", ["Developer", "User"])

temp_dir = tempfile.TemporaryDirectory()

# ----------------- DEVELOPER MODE -----------------
if mode == "Developer":
    st.subheader("Developer Mode")
    hidden_message = st.text_area("Masukkan hidden message yang ingin disisipkan ke QR Code")

    if st.button("Generate QR Code"):
        if hidden_message.strip() == "":
            st.warning("Pesan tidak boleh kosong!")
        else:
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(hidden_message)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")

            qr_path = f"{temp_dir.name}/qr_code.png"
            img.save(qr_path)

            st.image(qr_path, caption="QR Code Anda", use_column_width=True)
            with open(qr_path, "rb") as f:
                st.download_button("💾 Download QR Code", f, file_name="hidden_message_qr.png")

# ----------------- USER MODE -----------------
elif mode == "User":
    st.subheader("📷 User Mode - AR Camera Scanner (Real-time)")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            qr_detector = cv2.QRCodeDetector()
            data, bbox, _ = qr_detector.detectAndDecode(img)

            if bbox is not None and data:
                pts = np.int32(bbox).reshape(-1, 2)

                for i in range(len(pts)):
                    cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)

                top_center = np.mean([pts[0], pts[1]], axis=0).astype(int)
                x, y = top_center
                y -= 30

                overlay = img.copy()
                text = f"{data}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_w, text_h = text_size
                box_coords = ((x - text_w // 2 - 10, y - text_h - 10), (x + text_w // 2 + 10, y + 10))
                cv2.rectangle(overlay, box_coords[0], box_coords[1], (0, 0, 0), -1)
                img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
                cv2.putText(img, text, (x - text_w // 2, y),
                            font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="realtime-qrcode", video_processor_factory=VideoProcessor)
