# OPG Lens 🦷

A phone-friendly learning and second-look tool for dental panoramic radiographs (OPGs), by Dr. Swapnil Agrawal (BDS).

> **Not a diagnosis.** OPG Lens is a learning aid for students and a second-look aid for clinicians. It is not a medical device. Always confirm findings clinically and with appropriate radiographs.

## Two modes, one engine
- **Second look**: upload an OPG and get caries, deep caries, periapical lesions and impacted teeth boxed with FDI tooth numbers, plus a plain-language patient report in English and Hindi. Runs fully in the browser (onnxruntime-web), so the X-ray never leaves the device.
- **Learn**: the OPG of the Day (plus practice cases). Tap what you spot, name the finding and FDI number, then check against expert labels, with an explanation card for each finding. A daily streak is kept on your device.
- **Bring your own AI (optional)**: plug in any OpenAI-compatible base URL, key and model for a tutor chat. The key stays in your browser. Sending the image to your model is opt-in.

## Models (v0)
Two YOLO11s detectors at 1024 px, trained on a Kaggle T4 using the fully labelled DENTEX set (705 OPGs, 10% held out).

| Model | mAP50 | mAP50-95 |
|---|---|---|
| Findings (4 classes) | 0.557 | 0.381 |
| FDI number of abnormal teeth (32 classes) | 0.518 | 0.357 |

Per finding (mAP50): impacted 0.73, periapical lesion 0.57, deep caries 0.50, caries 0.43. The training script is in `training/train_v0.py`.

## Data and license
Trained on **DENTEX** (Hamamci et al., *DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays*, 2023), released under **CC BY-NC-SA 4.0**. The sample cases in `samples/` are DENTEX images. Because of this, the models and samples are for **non-commercial use only** and are shared under CC BY-NC-SA 4.0.
