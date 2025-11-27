import cv2
import streamlit as st
import numpy as np
import time

cv2.setNumThreads(1)

def main():

    st.title("Width Measurement via DroidCam")
    
    ip_camera_url = "http://192.168.137.30:4747/video"
    cap = cv2.VideoCapture(ip_camera_url)

    if not cap.isOpened():
        st.error("Unable to open DroidCam stream.")
        return
    
    pxl_ratio = 16 / 160
    stframe = st.empty()

    while True:
        ret, img = cap.read()
        if not ret:
            st.warning("Failed to get frame from DroidCam.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV)

        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 1000
        black_dots = [c for c in cnts if cv2.contourArea(c) > min_area]

        cv2.drawContours(img, black_dots, -1, (0, 255, 0), 2)

        for row_num in range(img.shape[0] - 1):
            row = gray[row_num: row_num + 1, :]
            left_px = np.argmax(row)
            row_rev = np.flip(row)
            right_px = img.shape[1] - np.argmax(row_rev)
            thickness = 2

            if row_num % 100 == 0 and left_px != 0 and right_px != 0:
                cv2.line(img, (left_px, row_num), (right_px, row_num), (0, 0, 255), thickness)
                distance = round((right_px - left_px) * pxl_ratio, 2)
                cv2.putText(img, f"{distance}mm", (left_px, row_num - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        stframe.image(img, channels="BGR")
        time.sleep(0.03)

if __name__ == "__main__":
    main()
