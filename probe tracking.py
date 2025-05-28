import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import logging
import os
import time
import pyzed.sl as sl
from threading import Thread

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
parameters = aruco.DetectorParameters_create()
marker_length = 0.025  # Marker size in meters

camera_matrix = np.load("C:/project_data11/calibration_images1_zed/camera_matrix.npy")
dist_coeffs = np.load("C:/project_data11/calibration_images1_zed/dist_coeffs.npy")

# CSV file for ArUco marker data
csv_file_path = "C:/project_data11/detected_markers1.csv"
csvfile = open(csv_file_path, mode="w", newline="")
csv_writer = csv.writer(csvfile)
csv_writer.writerow(
    [
        "Frame Count",
        "Marker ID",
        "Position X",
        "Position Y",
        "Position Z",
        "Orientation X",
        "Orientation Y",
        "Orientation Z",
    ]
)

zed = sl.Camera()
init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD1080, camera_fps=30)
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    logging.error("Unable to open ZED camera")
    exit(1)

runtime_params = sl.RuntimeParameters()
frame_count = 0
output_folder = "C:/project_data11/aruco+US_frames"
os.makedirs(output_folder, exist_ok=True)

capture_card_index = 1  # Change this to the appropriate device index for your capture card
ultrasound_cap = cv2.VideoCapture(capture_card_index)
if not ultrasound_cap.isOpened():
    logging.error("Unable to open ultrasound capture card")
    exit(1)

ultrasound_folder = "C:/project_data11/ultrasound_images"
os.makedirs(ultrasound_folder, exist_ok=True)


def capture_ultrasound_image(frame_id):
    ret, frame = ultrasound_cap.read()
    if ret:
        image_path = os.path.join(ultrasound_folder, f"ultrasound_{frame_id}.png")
        cv2.imwrite(image_path, frame)
        logging.info(f"Saved ultrasound image: {image_path}")
    else:
        logging.warning(f"Failed to capture ultrasound image for frame {frame_id}")


def capture_and_process_frames():
    global frame_count
    point_cloud = sl.Mat()

    while True:
        logging.info(f"Position the camera for frame {frame_count + 1}. Capturing in 5 seconds...")
        time.sleep(3)  # Add a 5-second gap

        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(point_cloud, sl.VIEW.LEFT)

            frame = point_cloud.get_data().copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            if ids is not None:
                logging.info(f"Detected markers: {ids.flatten()}")
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

                for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                    csv_writer.writerow(
                        [frame_count, ids[i][0], tvec[0][0], tvec[0][1], tvec[0][2], rvec[0][0], rvec[0][1], rvec[0][2]]
                    )
                csvfile.flush()

                frame_path = os.path.join(output_folder, f"frame_{frame_count}.png")
                cv2.imwrite(frame_path, frame[:, :, :3])
                logging.info(f"Saved frame {frame_count} with markers to: {frame_path}")
            else:
                logging.info(f"No markers detected in frame {frame_count}. Skipping...")

            ultrasound_thread = Thread(target=capture_ultrasound_image, args=(frame_count,))
            ultrasound_thread.start()

            frame_count += 1
        else:
            logging.warning("Failed to capture frame. Retrying...")

        if frame_count >= 100:
            break


try:
    capture_and_process_frames()
except KeyboardInterrupt:
    logging.info("Keyboard interrupt detected. Stopping capture.")
finally:
    zed.close()
    ultrasound_cap.release()
    csvfile.close()
    logging.info("Extrinsic calibration session ended.")

