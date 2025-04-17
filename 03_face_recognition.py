import cv2
import numpy as np
import os 

# Tạo bộ nhận diện khuôn mặt
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Kiểm tra xem mô hình có được tải không
if not os.path.isfile('trainer/trainer.yml'):
    raise FileNotFoundError("Model file not found.")
recognizer.read('trainer/trainer.yml')

# Đường dẫn tới tệp Haar cascade
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Khởi tạo bộ đếm ID
id = 0
names = ['None', 'Quynh', '', 'Carl', 'Joseph']  # Đảm bảo có đủ tên cho ID

# Khởi động video từ camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Thiết lập chiều rộng video
cam.set(4, 480)  # Thiết lập chiều cao video

# Định nghĩa kích thước tối thiểu để nhận diện khuôn mặt
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()  # Đọc khung hình từ camera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển đổi sang ảnh xám

    # Phát hiện khuôn mặt
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    if len(faces) == 0:  # Nếu không phát hiện khuôn mặt, tiếp tục vòng lặp
        continue

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ hình chữ nhật quanh khuôn mặt

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])  # Nhận diện khuôn mặt

        if confidence < 100:  # Nếu độ tin cậy cao
            id = names[id] if id < len(names) else "unknown"  # Kiểm tra ID hợp lệ
        else:
            id = "unknown"  # Nếu không nhận diện được

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)  # Hiển thị tên lên ảnh

    cv2.imshow('camera', img)  # Hiển thị video

    k = cv2.waitKey(10) & 0xff  # Chờ phím nhấn
    if k == 27:  # Nhấn ESC để thoát
        break

# Thông báo thoát chương trình và dọn dẹp
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()  # Giải phóng camera
cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ OpenCV