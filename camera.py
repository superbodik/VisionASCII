import cv2

def open_camera(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise Exception("Не удалось открыть камеру")
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def close_camera(cap):
    cap.release()
    cv2.destroyAllWindows()

def apply_zoom(frame, zoom_factor=1.0):
    if zoom_factor <= 1.0:
        return frame
    height, width = frame.shape[:2]
    new_w = int(width / zoom_factor)
    new_h = int(height / zoom_factor)
    x1 = (width - new_w) // 2
    y1 = (height - new_h) // 2
    cropped = frame[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (width, height))
