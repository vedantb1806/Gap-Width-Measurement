import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
import numpy as np
import logging
import os
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

# Additional imports from streamlit_webrtc
from streamlit_webrtc import webrtc_streamer


from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers



logger = logging.getLogger(__name__)

st.title("Width Measurement")

# Get ICE servers
ice_servers = get_ice_servers()
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:  # Changed av.videoframe to av.VideoFrame
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Corrected cv2.cvtcolor to cv2.cvtColor and cv2.COLOR_BGR2GRAY
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV)  # Corrected cv2.thresh_binary_inv to cv2.THRESH_BINARY_INV
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Corrected cv2.findcontours to cv2.findContours, cv2.RETR_EXTERNAL, and cv2.CHAIN_APPROX_SIMPLE
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 1000
    black_dots = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:
            black_dots.append(c)
    contours_only = np.zeros_like(img)  # Changed "frame" to "img" and added missing "np."

    cv2.drawContours(contours_only, black_dots, -1, (0, 255, 0), 2)  # Corrected cv2.drawcontours to cv2.drawContours and added thickness parameter

    pxl_ratio = 16 / 160
    for row_num in range(img.shape[0] - 1):
        row = gray[row_num: row_num + 1, :]
        left_px = np.argmax(row)
        row = np.flip(row)
        right_px = img.shape[1] - np.argmax(row)
        thickness = 2
        if row_num % 100 == 0 and left_px != 0 and right_px != 0:
            cv2.line(img, (left_px, row_num), (right_px, row_num), (0, 0, 255), thickness)
            distance = round((right_px - left_px) * pxl_ratio, 2)
            l_org = row_num - 10
            cv2.putText(img, str(distance) + "mm", (left_px, l_org), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Corrected cv2.puttext to cv2.putText and added missing cv2.
            print(distance)

    return av.VideoFrame.from_ndarray(img, format="bgr24")  # Changed av.videoframe to av.VideoFrame and "frame" to "img"

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)

st.markdown(
    "This demo uses a model and code created by Vedant Bajaj."
    "Under the Guidance of Mr. Arvind Jain."
    "Many thanks to Tata Motors Ltd."
)
