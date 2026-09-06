# 🍱 Food & Dish Recognition: YOLOv5 Object Detection & Instance Segmentation (Lab 5)

โปรเจกต์ตรวจจับและจำแนกประเภทอาหาร (Food Classification, Object Detection & Instance Segmentation) ด้วย **YOLOv5** และ **YOLOv5-seg** รองรับทั้งการระบุตำแหน่งด้วยกรอบสี่เหลี่ยม (Bounding Box) และการแยกส่วนภาพตามขอบเขตจริงของวัตถุ (Segmentation Mask)

---

## 📌 สารบัญ (Table of Contents)
- [ภาพรวมของโปรเจกต์ (Overview)](#-ภาพรวมของโปรเจกต์-overview)
- [โครงสร้างโปรเจกต์ (Project Structure)](#-โครงสร้างโปรเจกต์-project-structure)
- [คลาสในชุดข้อมูล (Dataset & Classes)](#-คลาสในชุดข้อมูล-dataset--classes)
- [การติดตั้งและเตรียมสภาพแวดล้อม (Installation & Setup)](#-การติดตั้งและเตรียมสภาพแวดล้อม-installation--setup)
- [ขั้นตอนการเทรนโมเดล (Model Training)](#-ขั้นตอนการเทรนโมเดล-model-training)
  - [1. Box Detection (YOLOv5)](#1-box-detection-yolov5)
  - [2. Instance Segmentation (YOLOv5-seg)](#2-instance-segmentation-yolov5-seg)
- [การติดตามและวัดผลลัพธ์ (Monitoring & Evaluation)](#-การติดตามและวัดผลลัพธ์-monitoring--evaluation)
- [การทำนายผลและการทดสอบ (Inference & Overlay)](#-การทำนายผลและการทดสอบ-inference--overlay)
- [รายการส่งงาน (Lab Checklist)](#-รายการส่งงาน-lab-checklist)

---

## 🔍 ภาพรวมของโปรเจกต์ (Overview)

โปรเจกต์นี้ถูกออกแบบมาเพื่อศึกษาและประยุกต์ใช้งานโมเดล Deep Learning สำหรับงานทางด้าน Computer Vision ในกลุ่มอาหารและภาชนะ โดยมีเป้าหมายหลัก 2 ด้าน:
1. **Object Detection (Box)**: ตรวจจับและระบุตำแหน่งของอาหารแต่ละชนิดด้วยกรอบ Bounding Box
2. **Instance Segmentation (Mask)**: สกัดมาสก์ความละเอียดระดับพิกเซล เพื่อแยกขอบเขตและรูปร่างของอาหารแต่ละชิ้น (Instance Mask)

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```text
Food class/
├── Lab5_YOLOv5_Box_Seg.ipynb       # โน้ตบุ๊กหลักสำหรับเทรน วัดผล และวิเคราะห์
├── test.py                         # สคริปต์ตรวจสอบ GPU/CUDA Environment
├── yolov5nu.pt                     # Pretrained Weights YOLOv5 Nano
├── yolov5s-seg.pt                  # Pretrained Weights YOLOv5s Segmentation
├── dataset/                        # ชุดข้อมูลอาหาร (Food Dataset)
│   ├── data.yaml                   # ไฟล์คอนฟิก Dataset, Paths, และ Classes
│   ├── images/                     # โฟลเดอร์รูปภาพ
│   │   ├── train/                  # ภาพสำหรับ Train
│   │   └── val/                    # ภาพสำหรับ Validation
│   ├── labels/                     # โฟลเดอร์ Labels (Polygon & Box coordinates)
│   │   ├── train/
│   │   └── val/
│   ├── train.txt                   # รายการพาธไฟล์รูปภาพ Train
│   └── val.txt                     # รายการพาธไฟล์รูปภาพ Validation
├── datasets/                       # ข้อมูลเสริม (เช่น COCO128)
├── runs/                           # โฟลเดอร์ผลลัพธ์จากการเทรนและการทดสอบ
│   └── train_seg/
│       └── exp/                    # ผลการเทรน Segmentation ล่าสุด
│           ├── weights/            # โมเดลที่บันทึก (best.pt, last.pt)
│           ├── results.csv         # ค่า Metric แต่ละ Epoch
│           ├── results.png         # กราฟ Loss, Precision, Recall, mAP
│           ├── confusion_matrix.png
│           ├── MaskPR_curve.png
│           └── BoxPR_curve.png
└── README.md                       # รายละเอียดคู่มือโปรเจกต์
```

---

## 🏷️ คลาสในชุดข้อมูล (Dataset & Classes)

โมเดลถูกฝึกให้รู้จักวัตถุและอาหารทั้งหมด 14 คลาส (แบ่งเป็น Mask Polygon และ Bounding Box `_bb`):

| Class ID | Class Name | คำอธิบาย |
|:---:|:---|:---|
| `0` | `eggs` | ไข่ดาว / ไข่ต้ม (Mask) |
| `1` | `dessert` | ของหวาน / ขนมหวาน (Mask) |
| `2` | `drink` | เครื่องดื่ม / แก้วน้ำ (Mask) |
| `3` | `spoonandfork` | ช้อนและส้อม (Mask) |
| `4` | `protein` | แหล่งโปรตีน / เนื้อสัตว์ (Mask) |
| `5` | `vegetable` | ผักชนิดต่างๆ (Mask) |
| `6` | `carb` | คาร์โบไฮเดรต เช่น ข้าว / เส้น (Mask) |
| `7` | `eggs_bb` | กรอบ Bounding Box: ไข่ |
| `8` | `dessert_bb` | กรอบ Bounding Box: ขนม |
| `9` | `drink_bb` | กรอบ Bounding Box: เครื่องดื่ม |
| `10` | `spoonandfork_bb` | กรอบ Bounding Box: ช้อนส้อม |
| `11` | `protein_bb` | กรอบ Bounding Box: โปรตีน |
| `12` | `vegetable_bb` | กรอบ Bounding Box: ผัก |
| `13` | `carb_bb` | กรอบ Bounding Box: คาร์โบไฮเดรต |

---

## ⚙️ การติดตั้งและเตรียมสภาพแวดล้อม (Installation & Setup)

### 1. ตรวจสอบสภาพแวดล้อมฮาร์ดแวร์ / GPU
รันสคริปต์ทดสอบ CUDA:
```bash
python test.py
```
*ตัวอย่างเอาต์พุต:*
```text
=== Device Check ===
CUDA Available: True
Number of GPUs: 1
GPU 0: NVIDIA GeForce RTX ...
====================
```

### 2. ติดตั้ง Dependencies และ Clone YOLOv5
```bash
# ติดตั้ง PyTorch รองรับ CUDA (ตัวอย่างสำหรับ CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ติดตั้งแพ็กเกจที่จำเป็น
pip install opencv-python matplotlib pandas seaborn scikit-learn PyYAML tensorboard tqdm

# Clone YOLOv5 repository (หากยังไม่มีใน workspace)
git clone https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
```

---

## 🚀 ขั้นตอนการเทรนโมเดล (Model Training)

### 1. Box Detection (YOLOv5)
ใช้โมเดลพื้นฐาน `yolov5s.pt` เพื่อตรวจจับแบบ Bounding Box:
```bash
python yolov5/train.py \
    --img 640 \
    --batch 16 \
    --epochs 50 \
    --data dataset/data.yaml \
    --weights yolov5s.pt \
    --project runs/train_box \
    --name exp \
    --exist-ok
```

### 2. Instance Segmentation (YOLOv5-seg)
ใช้โมเดล Segmentation `yolov5s-seg.pt` เพื่อทำนายทั้ง Bounding Box และ Segmentation Mask:
```bash
python yolov5/segment/train.py \
    --img 640 \
    --batch 8 \
    --epochs 50 \
    --data dataset/data.yaml \
    --weights yolov5s-seg.pt \
    --project runs/train_seg \
    --name exp \
    --exist-ok
```

> **หมายเหตุ:** ปรับ `--batch` และ `--img` ตามความจุ Memory ของ GPU ที่ใช้งาน

---

## 📊 การติดตามและวัดผลลัพธ์ (Monitoring & Evaluation)

### 1. เรียกดู TensorBoard
เปิด Dashboard สำหรับติดตาม Loss (Box Loss, Seg Loss, Obj Loss, Cls Loss) และ Learning Rate แบบเรียลไทม์:
```bash
tensorboard --logdir runs
```
จากนั้นเปิดเบราว์เซอร์ไปที่ `http://localhost:6006`

### 2. ผลการประเมิน (Validation / Test)
ประเมินคุณภาพโมเดลหลังการเทรน (Validation):
```bash
python yolov5/segment/val.py \
    --weights runs/train_seg/exp/weights/best.pt \
    --data dataset/data.yaml \
    --img 640 \
    --task val \
    --save-json
```

ไฟล์สรุปผลจะถูกบันทึกไว้ใน `runs/train_seg/exp/`:
- `results.png` : กราฟแสดงแนวโน้ม Loss และ Metrics ตลอดระยะเวลาการเทรน
- `BoxPR_curve.png` & `MaskPR_curve.png` : กราฟ Precision-Recall สำหรับ Bounding Box และ Mask
- `confusion_matrix.png` : เมทริกซ์ความแม่นยำในการจำแนกแต่ละคลาส

---

## 🎯 การทำนายผลและการทดสอบ (Inference & Overlay)

ทดสอบรันการทำนายผลบนภาพตัวอย่าง ด้วยน้ำหนักโมเดลที่ดีที่สุด (`best.pt`):

```bash
python yolov5/segment/predict.py \
    --weights runs/train_seg/exp/weights/best.pt \
    --source dataset/images/val \
    --img 640 \
    --conf 0.25 \
    --project runs/predict \
    --name exp \
    --exist-ok
```

ผลลัพธ์จะแสดงภาพพร้อมการ Overlay Bounding Box และ Segmentation Mask ของอาหารแต่ละจานและวัตถุอย่างชัดเจน

---

## ✅ รายการส่งงาน (Lab Checklist)

- [x] โคลนและจัดระเบียบ Repository พร้อมโครงสร้าง Dataset
- [x] ตรวจสอบความพร้อมของ GPU และ CUDA
- [x] รันเทรนโมเดล YOLOv5 Object Detection (Box)
- [x] รันเทรนโมเดล YOLOv5-seg Instance Segmentation (Mask)
- [x] บันทึกและแสดงกราฟ Loss, Learning Rate และ mAP@0.5 ผ่าน TensorBoard / `results.png`
- [x] สรุปค่า mAP@0.5 และกราฟ Precision-Recall Curve (Box & Mask)
- [x] บันทึกผลลัพธ์ Overlay (Bounding Boxes + Segmentation Masks) บนชุดข้อมูลทดสอบ
