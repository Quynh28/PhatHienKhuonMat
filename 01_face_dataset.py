import cv2
import os

# Mở camera (0 là camera mặc định)
cam = cv2.VideoCapture(0)
cam.set(3, 640) # Thiết lập rộng 640 pixel
cam.set(4, 480) # Thiết lập cao 480 pixel

# Tải mô hình nhận diện khuôn mặt từ file xml
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Nhập ID người dùng từ bàn phím
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Số lượng ảnh đã chụp
count = 0 

# Vòng lặp vô hạn để liên tục đọc hình ảnh từ camera
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, 1) # Lật ảnh theo chiều ngang để phản chiếu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Lặp qua tất cả các khuôn mặt được phát hiện
    for (x,y,w,h) in faces:
        # Vẽ hình chữ nhật quanh khuôn mặt được phát hiện
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1 # Tăng biến đếm số ảnh đã chụp

        # Lưu ảnh khuôn mặt vào thư mục với tên định dạng cụ thể
        cv2.imwrite("dataset/User ." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(300) & 0xff # Thời gian chờ đơn vị mili giây
    if k == 27: # Ấn ESC để dừng
        break
    elif count >= 200: # Khi đạt đủ số ảnh thì dừng
         break

# Thông báo thoát ctrình và dọn dẹp
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

"""
Sử dụng OpenCV để nhận diện khuôn mặt và lưu ảnh vào thư mục
Đoạn mã này mở camera, phát hiện khuôn mặt và lưu ảnh khuôn mặt vào thư mục với ID người dùng đã nhập.
Chương trình sẽ tiếp tục chạy cho đến khi người dùng nhấn phím ESC hoặc đã chụp đủ 200 ảnh.
"""