from ultralytics import YOLO
import cv2
model_api_kecil = YOLO('best_api_kecil.pt')
model_api_besar = YOLO('api_besar_model.pt')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model_api_kecil(frame, imgsz=640, conf=0.7)
    if any(len(result.boxes) > 0 for result in results):
        label = "API KECIL"
    else:
        results = model_api_besar(frame, imgsz=640, conf=0.5)
        label = "API BESAR" if any(len(result.boxes) > 0 for result in results) else "TIDAK ADA API"
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Campus Fire Guard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
