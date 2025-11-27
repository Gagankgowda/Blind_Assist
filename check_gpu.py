import torch
from ultralytics import YOLO

# 🔹 Check if GPU is available
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# 🔹 Load YOLO model
model = YOLO(r"C:\Users\Gagan\OneDrive\Desktop\python_3.11\runs\detect\train\weights\best.pt")

# 🔹 Run inference on GPU
results = model("temp.jpg", device=0)  # 0 = first GPU
print("Model is running on:", model.device)

# Show result
results.show()
