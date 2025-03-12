from ultralytics import YOLO
import os

model_api_kecil = YOLO('best_api_kecil.pt')
model_api_besar = YOLO('api_besar_model.pt')


image_path = "FirePhotography.jpg"


output_dir = "runs/detect/"
os.makedirs(output_dir, exist_ok=True)


results_api_kecil = model_api_kecil.predict(source=image_path, imgsz=640, conf=0.5)

if any(len(result.boxes) > 0 for result in results_api_kecil):
    print("Terdeteksi api kecil, menyimpan hasil dari model api kecil.")
    model_api_kecil.predict(source=image_path, imgsz=640, conf=0.5, save=True)
else:
    print("Tidak ada api kecil, menjalankan model api besar.")
    model_api_besar.predict(source=image_path, imgsz=640, conf=0.5, save=True)

print(f"Gambar hasil prediksi disimpan di folder: {output_dir}")
