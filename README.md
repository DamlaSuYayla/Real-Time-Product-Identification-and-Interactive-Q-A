# Real-Time Product Identification and Interactive Q&A Using Quantised YOLOv8-Nano on Edge Devices for Visually Impaired Accessibility

An entirely offline, privacy-first Edge AI solution built to empower visually impaired individuals to identify retail products independently. The framework combines deep learning-based barcode localization, structured model pruning, a localized relational database, and multimodal audio feedback.

---

## 📌 Project Overview
Independent shopping presents severe navigation and accessibility challenges for the visually impaired population (VIP). While commercial packaging contains barcodes, extracting this data without cloud-dependent latency or internet reliance remains difficult.

This project implements a fully on-device, localized pipeline that:
* **Detects & Segments:** Uses an optimized YOLOv8-Nano network to isolate 1D barcodes under real-world distortions.
* **Decodes Locally:** Integrates a dual-path pipeline (fast-path via `pyzbar` and robust-path via padded YOLO cropping).
* **Retrieves Context:** Queries an offline, serverless SQLite database containing ~200,000 UK product records.
* **Interacts via Voice:** Features a rule-based QA engine triggered by voice commands to read out ingredients, allergens, and product descriptions using edge text-to-speech.

---

## 🏗️ End-to-End Pipeline Architecture
The system operates sequentially entirely on-device to comply with UK GDPR data minimization principles:
1. **Camera Input:** User uploads an image or utilizes a live camera feed via the interface.
2. **Dual-Path Decoding:** Direct full-image decode attempt $\rightarrow$ Fallback to YOLOv8 bounding box localization with a 35% padding ratio.
3. **Offline Lookup:** Querying the relational schema via zero-padded barcode primitives.
4. **Rule-Based QA Dispatcher:** Transcribing queries via `SpeechRecognition` and routing product metadata fields.
5. **Speech Synthesis Engine:** Delivering natural British English feedback (`en-GB-SoniaNeural`) via `edge-tts`.

---

## 📊 Experimental Results & Model Optimization

The network was fine-tuned for 50 epochs on the **InventBar & ParcelBar** datasets and compressed using global unstructured $L_1$-norm pruning (30% convolutional weights removed).

### Performance Benchmarking
Tested under controlled CUDA synchronization splits (100 measured cycles):

| Model Variant | Format | Model Size (MB) | Throughput (FPS) | Inference Latency (ms) | Peak Memory (MB) | mAP@0.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | PyTorch | 5.95 | **39.21** | **24.22 ± 3.23** | 1.86 | 0.984 |
| **Pruned (Optimized)** | PyTorch | 5.94 | 35.43 | 26.61 ± 4.18 | **1.86** | 0.984 |
| **FP16** | ONNX | 5.86 | 7.94 | 122.76 ± 23.73 | 7.13 | 0.985 |
| **FP32** | ONNX | 11.67 | 7.19 | 135.35 ± 80.25 | 9.48 | 0.985 |
| **INT8** | ONNX | **3.29** | 10.34 | 94.93 ± 12.40 | 9.47 | 0.000 (Failed) |

### Key Optimization Takeaways:
* **Hypothesis Validation:** All operational variants met the testable hypothesis threshold of $\ge15$ FPS with $<10\%$ accuracy loss.
* **The Pruned Model** achieved a 0.984 $mAP_{50}$, preserving near-identical accuracy (only 0.03% drop) with the lowest peak memory utilization.
* **INT8 Failure:** Static INT8 quantization failed ($mAP_{50}=0.000$) due to calibration discrepancies on MinMax threshold splits.

---

## 🛠️ Tech Stack & Frameworks
* **Computer Vision:** `Ultralytics YOLOv8-Nano`, `OpenCV`, `pyzbar`
* **Edge Execution & UI:** `Streamlit` (WCAG 2.1 Compliant), `ONNX Runtime`, `SQLite3`
* **Speech Layers:** `edge-tts`, `SpeechRecognition`

---

## 🚀 How to Run (Streamlit App)

Notebook link: 
https://colab.research.google.com/drive/1oRCz3R4UoBfl7d71O_G8E1AS2BfSYLTj?usp=sharing
