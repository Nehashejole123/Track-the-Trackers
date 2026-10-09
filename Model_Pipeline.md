# 🐅 TigerID — Machine Learning Model Pipeline & System Contracts
> **Document Code:** `MODEL_PIPELINE.md`  
> **Status:** `ACTIVE SPECIFICATION` | **Target Platform:** Offline Workstation & Mobile Field Companion  
> **Target Reserve:** Tadoba Andhari Tiger Reserve (TATR) & SynoraAI Benchmarks

---

## 🎬 Part 1: How the "Matkasur" Live Demo Actually Works

### 1. 📥 Offline Preparation (Enrolling Matkasur into the Database)

```text
Reference Photo of Matkasur (e.g., from web / park archive)
                   │
                   ▼
       [ Re-ID Feature Extractor ]
                   │
                   ▼
Vector Embedding: [0.14, -0.82, 0.45, ... 512 floats]
                   │
                   ▼
Saved in SQLite / FAISS:
Row: { id: "T-54", name: "Matkasur", flank: "LEFT", vector: [...] }
```

---

### 2. 🔍 Live Test 1 (Recognizing Matkasur)

```text
New Query Image: Different, angled, or nighttime photo of Matkasur
                   │
                   ▼
       [ Re-ID Feature Extractor ]
                   │
                   ▼
Query Vector: [0.15, -0.80, 0.43, ... 512 floats]
                   │
                   ▼
Cosine Similarity vs SQLite Database:
├── Match vs T-54 (Matkasur) = 0.91 (91%)
├── Match vs All other tigers < 0.40
└── Outcome: "Match Confirmed: T-54 (Matkasur) | Confidence: 91%"
```

---

### 3. ❓ Live Test 2 (Unregistered Tiger)

```text
New Query Image: A photo of an un-enrolled tiger (or another reserve's tiger)
                   │
                   ▼
       [ Re-ID Feature Extractor ]
                   │
                   ▼
Query Vector: [-0.62, 0.11, -0.29, ... 512 floats]
                   │
                   ▼
Cosine Similarity vs SQLite Database:
├── Max Similarity found across all entries = 0.48 (< 0.75 threshold)
├── Outcome: "No Match Found | Flagged: UNKNOWN_NEW"
└── One-Click UI Button: "Register as T-116?" (Appends vector to DB instantly)
```

> [!TIP]
> **Hackathon Demonstration Strategy:**  
> For the hackathon demo, take 2–3 public photos of Matkasur from the web and run them through your extractor to pre-populate his profile in your SQLite database. When you run a 4th unseen photo during the presentation, the math matches automatically without any custom model training.

---

## 📐 Part 2: Technical Specifications & Contracts

This document defines the machine learning contracts, tensor dimensions, preprocessing requirements, operational thresholds, and export specifications for the **TigerID** system.

### 🔄 End-to-End Visual Dataflow

```text
[ Input Image (1920x1080) ]
             │
             ▼
┌─────────────────────────┐
│ MODEL 1: YOLOv8 Detector│ ──► Bounding Box [x1, y1, x2, y2, conf, class]
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Heuristic Quality Gate  │ ──► Drops blurred / blown-out frames
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Flank Orientation Gate  │ ──► Classifies: Left (0) / Right (1)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ MODEL 2: Re-ID Feature  │ ──► 512-Dimensional L2-Normalized Vector
│ Extractor (ResNet-50)   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Vector Metric Search    │ ──► Cosine Similarity vs SQLite/FAISS Database
└─────────────────────────┘
```

---

### 1. 🎯 Model 1: Animal Detector (YOLOv8)

#### 📋 Specifications
* **Base Architecture:** YOLOv8n (Nano) or YOLOv8s (Small) fine-tuned on wildlife camera trap data.
* **Input Tensor:** $[1, 3, 640, 640]$ (Batch, Channels, Height, Width).
* **Color Space:** RGB, normalized to $[0.0, 1.0]$.
* **Target Classes:** `Class 0: tiger` (all background classes, non-target wildlife, and false PIR triggers are ignored).

#### 📤 Output Contract
* **Raw Output Tensor:** $[1, 5, 8400]$ (representing coordinate proposals and class probabilities).
* **Post-NMS Parsed Format:**
  $$\mathbf{B} = [x_1, y_1, x_2, y_2, \text{confidence}, \text{class\_id}]$$
  * $x_1, y_1, x_2, y_2$: Normalized coordinates in range $[0.0, 1.0]$ relative to original image dimensions.
  * $\text{confidence}$: Float in range $[0.0, 1.0]$.

#### ⚙️ Inference Parameters
* **Confidence Threshold ($\tau_{\text{conf}}$):** $0.50$
* **Non-Maximum Suppression IoU Threshold ($\tau_{\text{IoU}}$):** $0.45$

---

### 2. 🧪 Pre-Extraction Processing & Quality Gating

Before passing cropped tiger regions to the feature extractor, two validation checks are executed to prevent false embeddings from entering the vector database:

#### A. 🔍 Blur Detection (Laplacian Variance)
Converts the cropped bounding box to grayscale and computes the Laplacian operator variance $\sigma^2_{\text{Lap}}$:

$$\sigma^2_{\text{Lap}} = \operatorname{Var}(\nabla^2 I)$$

> [!WARNING]
> **Quality Gate Condition:**  
> If $\sigma^2_{\text{Lap}} < 80.0$, the image is classified as **motion-blurred** and rejected from automatic identification.

#### B. ↔️ Flank Orientation Routing
Because tigers have asymmetric stripe patterns on their left and right flanks, each side is cataloged as a distinct entity.

* **Orientation Classifier:** Lightweight MobileNetV3 head.
* **Classes:**
  * `0`: **Left Flank** *(Routed to Left Flank Vector Index)*
  * `1`: **Right Flank** *(Routed to Right Flank Vector Index)*
  * `2`: **Frontal / Rear** *(Logged for population counts, skipped for stripe matching)*

---

### 3. 🧬 Model 2: Biometric Re-ID Extractor (Pose-Aware ResNet-50)

#### 📋 Specifications
* **Architecture:** ResNet-50 backbone with Regional Average Pooling (RAP) at layer `res3d` and Squeeze-and-Excitation attention head (**PPbM-b** architecture).
* **Input Tensor:** $[1, 3, 128, 256]$ (Batch, Channels, Height, Width) — optimized for quadruped horizontal aspect ratio ($2:1$).
* **Preprocessing:**
  * Resize to $256 \times 128$ ($\text{Width} \times \text{Height}$).
  * Pixel Normalization using ImageNet parameters:
    * $\text{Mean} = [0.485, 0.456, 0.406]$
    * $\text{Std} = [0.229, 0.224, 0.225]$

#### 📤 Output Contract
* **Raw Output:** 512-dimensional continuous feature representation $\mathbf{z} \in \mathbb{R}^{512}$.
* **$L_2$ Normalization:**
  $$\hat{\mathbf{v}} = \frac{\mathbf{z}}{\Vert{}\mathbf{z}\Vert{}_2} = \frac{\mathbf{z}}{\sqrt{\sum_{i=1}^{512} z_i^2}}$$
* **Export Storage:** Stored in SQLite as a raw binary `BLOB` ($512\text{ floats} \times 4\text{ bytes} = 2,048\text{ bytes}$).

---

### 4. 🧮 Vector Metric Search & Classification Logic

Let $\mathbf{q} \in \mathbb{R}^{512}$ be the $L_2$-normalized query vector extracted from an incoming photo, and let $\mathbf{g}_k \in \mathbb{R}^{512}$ be the $k$-th gallery vector stored in the database for the matching flank side.

#### 📏 Distance Metric
The affinity score is computed using **Cosine Similarity** (equivalent to the dot product of $L_2$-normalized vectors):

$$S(\mathbf{q}, \mathbf{g}_k) = \mathbf{q} \cdot \mathbf{g}_k = \sum_{i=1}^{512} q_i \cdot g_{k,i}$$

* **Cosine Similarity Bounds:** $S \in [-1.0, 1.0]$, where identical patterns approach $1.0$.

#### 🚦 Decision Boundary Protocol

| Metric Threshold | System Status | Triggered Action |
| :--- | :--- | :--- |
| `S_max ≥ 0.75` | 🟢 **`MATCH_CONFIRMED`** | Assigns existing `tiger_id`. Automatically appends record to Sighting Attendance Ledger. |
| `0.65 ≤ S_max < 0.75` | 🟡 **`MATCH_UNCERTAIN`** | Flags sighting for human-in-the-loop verification. Displays top-3 candidates side-by-side. |
| `S_max < 0.65` | 🔴 **`UNKNOWN_NEW`** | Flags animal as unregistered resident. Triggers one-click new profile enrollment modal. |

---

### 5. 📦 Model Serialization & Export Targets

To enable local execution on office PCs (without requiring GPUs) and mobile field devices, the PyTorch models are converted to optimized runtime formats.

#### 🖥️ Desktop / Workstation Targets
* **Format:** ONNX (Open Neural Network Exchange), Opset 17.
* **Execution Provider:** ONNX Runtime (`CPUExecutionProvider` / OpenVINO).
* **Target Latency:** $< 120\text{ ms}$ per tiger crop on an Intel Core i5/i7 processor.

#### 📱 Mobile Field Companion Targets
* **Format:** TensorFlow Lite (`.tflite`).
* **Quantization:** FP16 quantization for weights to preserve stripe resolution while cutting model size by 50%.
* **Size Limits:**
  * **Detector (`yolov8n.tflite`):** $\le 6.5\text{ MB}$
  * **Re-ID Extractor (`tiger_reid_fp16.tflite`):** $\le 45\text{ MB}$
* **Target Latency:** $< 350\text{ ms}$ on standard Android ARM64 processors.