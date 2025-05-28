# ultrasound-probe-tracking
# Ultrasound Probe Tracking with ArUco and ZED 2

This project provides a Python script for real-time tracking of an ultrasound probe using **ArUco markers**, a **ZED 2 stereo camera**, and **OpenCV**, while simultaneously capturing ultrasound images using a video capture card.

---

## 🎯 Objectives

- Detect ArUco markers on the ultrasound probe
- Estimate pose (position & orientation) using intrinsic parameters
- Capture and save synchronized ultrasound images
- Save 3D marker data to a CSV file for reconstruction

---

## 🔧 Setup

- **Camera**: ZED 2
- **Marker**: ArUco (DICT_4X4_1000), size: 0.025 m
- **Capture Device**: External ultrasound video grabber (set via `capture_card_index`)
- **Dependencies**:
  - Python
  - OpenCV
  - pyzed (Stereolabs SDK)
  - NumPy
  - Threading

---

## 🗂 File Structure

