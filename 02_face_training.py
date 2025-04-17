import cv2
import numpy as np
from PIL import Image
import os

# Đường dẫn cho csdl hình ảnh khuôn mặt
path = 'dataset'

# Tạo một đối tượng nhận diện khuôn mặt sử dụng LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Tải mô hình phát hiện khuôn mặt từ file xml
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Hàm để lấy dữ liệu hình ảnh và nhãn
def getImagesAndLabels(path):
    # Tạo danh sách đường dẫn hình ảnh trong thư mục
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[] # Danh sách để lưu ảnh khuôn mặt
    ids = [] # Danh sách để lưu ID tương ứng với mỗi ảnh

    for imagePath in imagePaths:
        # Mở ảnh và chuyển đổi sang ảnh xám
        PIL_img = Image.open(imagePath).convert('L') # chuyển đổi thành ảnh xám
        img_numpy = np.array(PIL_img,'uint8') # Chuyển đổi ảnh thành mảng NumPy

        # Lấy ID từ tên file
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        # Phát hiện khuôn mặt trong ảnh xám
        faces = detector.detectMultiScale(img_numpy)

        # Lặp qua tất cả các khuôn mặt đã phát hiện
        for (x,y,w,h) in faces:
            # Lưu ảnh khuôn mặt và ID vào danh sách
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    # Trả về danh sách khuôn mặt và ID
    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
# Gọi hàm để lấy dữ liệu hình ảnh và nhãn
faces,ids = getImagesAndLabels(path)
# Huấn luyện mô hình nhận diện khuôn mặt
recognizer.train(faces, np.array(ids))

# Lưu mô hình vào file trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # Lưu mô hình đã huấn luyện

# In ra số lượng khuôn mặt đã được huấn luyện và kết thúc chương trình
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

"""
Đoạn mã này sử dụng OpenCV để huấn luyện một mô hình nhận diện khuôn mặt.
Lấy dữ liệu từ các hình ảnh đã được lưu trong thư mục dataset,
phát hiện khuôn mặt trong từng hình ảnh và lưu mô hình vào một file YAML để sử dụng sau.
"""