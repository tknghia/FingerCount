import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import time

model = tf.keras.models.load_model('model.h5')
video = cv2.VideoCapture(0)
box_x, box_y, box_w, box_h = 100, 100, 300, 300
last_capture_time = time.time()
while True:
    ret, frame = video.read()  # Đọc khung hình từ webcam
    if not ret:
        break

    # Vẽ khung xanh lên khung hình
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("Capturing", frame)

    # Kiểm tra thời gian chụp
    current_time = time.time()
    if current_time - last_capture_time >= 2:  # Nếu đã qua 5 giây
        # Cắt phần khung xanh từ khung hình
        roi = frame[box_y:box_y + box_h, box_x:box_x + box_w]

        # Chuyển đổi khung hình từ BGR sang RGB
        im = Image.fromarray(roi, 'RGB')

        # Thay đổi kích thước ảnh
        im = im.resize((128, 128))

        # Chuyển ảnh PIL thành mảng NumPy
        img_array = np.array(im)

        # Chuẩn hóa giá trị pixel và thêm trục batch
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Dự đoán số lượng ngón tay
        prediction = int(np.argmax(model.predict(img_array)[0]))
        print(prediction)

        # Nếu dự đoán là 0, chuyển khung hình thành ảnh grayscale
        if prediction == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Nếu khung hình là grayscale, chuyển nó về ảnh màu để hiển thị văn bản màu
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Hiển thị số lượng ngón tay lên khung hình
        cv2.putText(frame, f'Finger Count: {prediction}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)

        # Cập nhật thời gian chụp ảnh gần nhất
        last_capture_time = current_time

    # Thoát khi nhấn phím 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
# Giải phóng tài nguyên
video.release()
cv2.destroyAllWindows()