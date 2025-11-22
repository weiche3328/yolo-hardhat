import cv2
from ultralytics import YOLO

# 類別順序要跟你訓練時的一致：0: helmet, 1: person, 2: head
CLASS_NAMES = ["helmet", "person", "head"]

# 用你剛剛 scp 下來的 best.pt
model = YOLO("best.pt")

# Mac 的內建鏡頭通常是 index 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 推論 (直接丟 numpy array 即可)
    results = model(frame, imgsz=640, conf=0.5)
    result = results[0]

    # 畫框
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    cv2.imshow("Hard Hat Detector (Mac)", frame)

    # 按 q 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
