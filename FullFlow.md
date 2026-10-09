# 🐅 TigerID Core & Field Platform
> **Master End-to-End System Architecture & Complete Flow Specification**  
> *Offline-First Biometric Re-Identification & Field Intelligence for Tadoba Andhari Tiger Reserve (TATR)*

---

## 📖 Overview

Building this production system requires a modular, multi-model pipeline. Below is the complete, end-to-end architecture and project breakdown designed for direct conversion into system specification documentation (`.md` files).

---

## 🏗️ System Architecture: TigerID Core & Field Platform

```text
[ Raw SD Card / Camera Trap Ingestion (500+ Images) ]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: SPECIES CULLING & IMAGE QUALITY ASSESSMENT (IQA)   │
│ • Model: YOLOv8x / MegaDetector-v5                         │
│ • Logic: Discard empty, foliage, non-target fauna           │
│ • Metric: Laplacian Variance (Blur) + Exposure Histograms   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Filtered to ~80 High-Quality Tiger Crops)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: BIOMETRIC PARSING & SKELETON POSE ALIGNMENT        │
│ • Orientation Net: Left Flank vs. Right Flank Classifier    │
│ • Pose Model: HRNet (15 Anatomical Keypoints)               │
│ • Decomposition: 7-Part AABB Generation (PPbM)             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: DEEP METRIC RE-ID MATCHING ENGINE                  │
│ • Feature Extractor: ResNet-50 Intermediate RAP (res3d)     │
│ • Attention Head: Squeeze-and-Excitation (PPbM-b)           │
│ • Inference: 512-D L2-Normalized Embedding Vector           │
│ • Search: Local FAISS Index / Cosine Similarity Engine       │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
[ Match Found: Cosine Sim ≥ 0.75 ]    [ Unregistered: Cosine Sim < 0.75 ]
  • Tiger ID: T-04                      • Flag as "Unknown_2026_X"
  • Sighting Attendance Logged          • Human-in-the-Loop Review Screen
            │                                     │
            └──────────────────┬──────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: EXEMPLAR SELECTION & GALLERY CURATION             │
│ • Rank images by: Composite Score = Sharpness × Alignment   │
│ • Keep Top-5 Exemplars in `tigers/{tiger_id}/gallery/`      │
│ • Write spatial log to SQLite / SECR-compatible CSV         │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 🤖 Multi-Model AI & Pipeline Specifications

### 🎯 Model 1: Species Detector & Bounding-Box Localizer
* **Base Architecture:** YOLOv8 (fine-tuned on wild fauna) or MegaDetector-v5.
* **Function:** Ingests raw $1920 \times 1080$ images. Detects target bounding boxes and drops empty background images (wind, swaying trees, false PIR triggers) and non-target species (cattle, wild boar, deer).
* **Runtime Optimization:** Quantized to INT8/FP16 ONNX format.

#### 🛡️ Quality Gating & Pre-Filtering Engine (Heuristic / Classical CV)
* **Laplacian Variance Metric:** Computes the variance of the Laplacian over the cropped tiger bounding box to calculate sharpness.
  $$\sigma^2_{\text{Laplacian}} < \tau_{\text{blur}}$$
  If the score falls below threshold $\tau_{\text{blur}}$, the image is routed to a secondary archive to prevent blurry features from corrupting embeddings.
* **Luminance Thresholding:** Rejects extreme flare or pitch-black IR sensor dropouts.

---

### 📐 Model 2: Flank Orientation & Keypoint Estimator

#### ↔️ Orientation Classifier
* **Architecture:** Lightweight CNN (MobileNetV3) classifying the cropped animal into:
  * `0`: **Left Flank**
  * `1`: **Right Flank**
  * `2`: **Frontal / Head-On** *(Ignored for stripe Re-ID)*
  * `3`: **Rear** *(Ignored)*
* **Rationale:** Tigers exhibit distinct, asymmetrical stripe patterns on their left and right sides. Left and right profiles must be managed as separate biometric entities.

#### 🦴 Pose Estimator (HRNet Backbone)
* **Training Base:** Fine-tuned on the ATRW benchmark (15 anatomical keypoints: ears, nose, shoulders, paws, hips, knees, tail root, center).
* **Outputs:** 15 $(x, y, v)$ keypoint coordinates where $v$ indicates visibility.

---

### 🧬 Model 3: Pose Part-Based Model (PPbM-b) Feature Extractor
* **Backbone:** ResNet-50 pre-trained backbone.
* **Regional Average Pooling (RAP):** Uses predicted skeleton coordinates to construct 7 Axis-Aligned Bounding Boxes (AABBs):
  1. Trunk
  2. Left Front Leg
  3. Right Front Leg
  4. Left Hind Thigh
  5. Right Hind Thigh
  6. Left Hind Shank
  7. Right Hind Shank  
  Extracts local pooling maps directly from the intermediate `res3d` layer to retain high spatial resolution ($32 \times 16$).

* **Soft-Attention Fusion:** Passes local features through a Squeeze-and-Excitation (SE) block:
  $$F_{i=1:7}[y] = \sum_{i=1}^{7} \alpha_i y_i$$
  Calculates attention weights $\alpha_i$ dynamically to down-weight obscured body parts (such as legs blocked by grass) and emphasize visible flank patterns.
* **Embedding Output:** 512-dimensional $L_2$-normalized vector embedding.

---

## 2. 🖥️ Desktop / Web Application: Biologist Workstation

The desktop system serves as the central analytical hub for wildlife biologists and reserve officers, operating **100% locally on standard office hardware**.

### 💻 Tech Stack
* **Backend Core:** Python (FastAPI), running locally.
* **Inference Engine:** ONNX Runtime / OpenVINO (CPU-accelerated execution without external GPU dependencies).
* **Frontend UI:** Flutter Desktop (Windows/Linux) or React + Tailwind CSS wrapped in Electron / Tauri.
* **Local Vector Store:** FAISS (Facebook AI Similarity Search) index flat file (`.index`) alongside SQLite database (`tatr_local.db`).

### 📊 Core Functional Modules Dashboard

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   TATR BIOLOGIST WORKSTATION DASHBOARD                 │
├───────────────────────┬────────────────────────────────────────────────┤
│ [📁 Batch SD Ingestion]│ Sighting Verification: 2026-10-08_1422.JPG     │
│                       ├──────────────────────┬─────────────────────────┤
│ Active Camera:        │ Ingested Image:      │ Matched Database ID:    │
│ [CT-KOLSA-08        ▼]│ [ Cropped Flank ]    │ [ T-04 (Left Flank)   ] │
│                       │                      │ Match Confidence: 94.6% │
│ Ingestion Progress:   │ Pose: 15 Pts Valid   │ Status: Confirmed Match │
│ [████████████ 100%]   ├──────────────────────┴─────────────────────────┤
│ Processed: 520        │ Action:                                        │
│ Tigers Detected: 42   │ [ ✅ Approve & Log ]  [ ❓ Mark Unknown ]      │
├───────────────────────┴────────────────────────────────────────────────┤
│ 🐯 Master Exemplar Gallery: Tiger T-04 (Matkasur)                       │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│ │ Best Flank 1 │ │ Best Flank 2 │ │ Night IR Best│ │ Territory: Kolsa│ │
│ │ (Sharp: 0.94)│ │ (Sharp: 0.91)│ │ (Sharp: 0.88)│ │ Sightings: 64   │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ └─────────────────┘ │
├────────────────────────────────────────────────────────────────────────┤
│ 🗺️ GIS Spatiotemporal Trajectory & Conflict Heatmap                     │
│ [ Interactive OpenStreetMap Layer showing camera nodes & move vectors ]│
├────────────────────────────────────────────────────────────────────────┤
│ 💬 Conservation RAG Query Console                                      │
│ > "List all sightings of T-04 logged after 18:00 within Kolsa range"    │
└────────────────────────────────────────────────────────────────────────┘
```

#### A. 📁 SD Card Bulk Pipeline & Filtering Ingestion
* Drag-and-drop ingestion of raw camera-trap folders.
* Maps Station IDs from folder paths (e.g., `/DCIM/CT_MOHARLI_14/`) to pre-indexed GPS coordinates stored in the local SQLite table.
* Displays batch metrics: Total photos ingested, blanks rejected, target tiger frames isolated, and matches computed.

#### B. 👤 Human-in-the-Loop Verification Console
* Pre-sorts matches:
  * **High-confidence:** $>85\%$
  * **Edge-cases:** $70\% - 85\%$
  * **Unregistered:** $<70\%$
* Displays side-by-side verification: Input camera-trap crop alongside the tiger's best database exemplar image with visual stripe overlay.
* One-click commitment updates the official sighting attendance log.

#### C. 🖼️ Automated Exemplar Gallery Manager
* Automatically curates a folder for each individual: `data/catalog/{tiger_id}/`.
* Maintains the **Top-5 canonical images** (e.g., 2 left-flank daytime, 2 left-flank nighttime IR, 1 full-body context).
* If a newly ingested photo achieves a higher sharpness/alignment score than an existing exemplar, the system updates the gallery profile automatically.

#### D. 🗺️ Trajectory Tracking & SECR Analytics
* Visualizes spatiotemporal trajectories: draws sequential movement lines connecting camera stations over time.
* Generates Minimum Convex Polygons (MCP) to compute territory areas in square kilometers.
* Exports directly to SECR-compliant formats (Spatially Explicit Capture-Recapture): `Session`, `IndividualID`, `Occasion`, `DetectorID`, `Sex`.

#### E. 🤖 Local Natural Language Query Agent (RAG)
* Embedded local language model or SQL agent (Llama-3.2-3B via `llama.cpp` or SQLite full-text search).
* Translates conversational queries into structured SQL operations to return sighting counts, last-known locations, and territory overlap metrics without internet access.

---

## 3. 📱 Mobile Application: Field Companion (Ranger Edition)

Designed for field patrols, Rapid Response Units (RRU), and snare-combing operations in areas with **zero cellular connectivity**.

### 📱 Tech Stack
* **Framework:** Flutter (Android APK target for rugged field smartphones).
* **Edge AI Engine:** TFLite (TensorFlow Lite) / ONNX Runtime Mobile.
* **Local Storage:** SQLite (via `sqflite`) storing vector tables and metadata locally.
* **Sensors:** Hardware GNSS (`geolocator`) accessing GPS/GLONASS/NavIC satellites directly without cellular data.

### 📱 Functional Workflows & UI Viewfinder

```text
┌──────────────────────────────────────────────────────────┐
│                   TATR RANGER FIELD COMPANION            │
├──────────────────────────────────────────────────────────┤
│ Mode Select:                                             │
│ [ 📷 Live Camera Capture ]   [ 💾 OTG Card Reader Mode ]  │
├──────────────────────────────────────────────────────────┤
│ Current Hardware Status:                                 │
│ GPS Fix: 20.2144° N, 79.3121° E (Accuracy: ±3.2m)        │
│ Network: OFFLINE (Airplane Mode Active)                  │
│ Local Database: 114 Registered Tigers Synced             │
├──────────────────────────────────────────────────────────┤
│ Field Capture Preview:                                   │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ [ Live Viewfinder with Tiger BBox & Orientation HUD] │ │
│ │ Status: Left Flank Detected                          │ │
│ └──────────────────────────────────────────────────────┘ │
│                                                          │
│ [ RUN OFFLINE IDENTIFICATION ]                           │
├──────────────────────────────────────────────────────────┤
│ Instant Result:                                          │
│ 🐅 MATCH: Tiger T-54 (Confidence: 91.2%)                 │
│ Alert: Registered Resident of Core Zone 3                │
│ Warning: Animal is outside standard territory boundary!   │
├──────────────────────────────────────────────────────────┤
│ Action:                                                  │
│ [ 💾 Save Sighting to Local Ledger ]   [ ⚠️ Log Incident ]│
└──────────────────────────────────────────────────────────┘
```

#### A. 🔌 OTG Camera-Trap Field Verification
* Allows guards to plug a camera-trap SD card directly into their smartphone using a USB-C OTG adapter while standing at the camera tree.
* Scans the card to confirm whether target conflict animals were photographed before leaving the beat.

#### B. ⚡ Conflict & Encounter Identification
* Allows rapid identification during direct encounters, carcass examinations, or conflict operations.
* Instant visual matching against stored local vectors indicates if the individual has an active conflict history.

#### C. 🛰️ Hardware GPS Auto-Tagging
* Extracts spatial coordinates from the phone's internal GNSS chip.
* Attaches coordinates, compass bearing, and satellite timestamps to the record, caching it locally for delta synchronization once the device returns to base camp.

---

## 4. 🗄️ Relational Database Schema (`tatr_local.db`)

```sql
-- Registered Tiger Master Table
CREATE TABLE tigers (
    tiger_id VARCHAR(32) PRIMARY KEY, -- e.g., 'T-04'
    common_name VARCHAR(64),          -- e.g., 'Matkasur'
    estimated_sex VARCHAR(8),         -- 'MALE', 'FEMALE', 'UNKNOWN'
    first_recorded_date DATE,
    status VARCHAR(16) DEFAULT 'ACTIVE' -- 'ACTIVE', 'MISSING', 'DECEASED'
);

-- Biometric Flank Profile (Separating Left and Right Entities)
CREATE TABLE biometric_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tiger_id VARCHAR(32) REFERENCES tigers(tiger_id),
    flank_side VARCHAR(8) NOT NULL,    -- 'LEFT', 'RIGHT'
    embedding_blob BLOB NOT NULL,      -- 512-D Float32 vector (2048 bytes)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Camera Trap Deployments
CREATE TABLE camera_stations (
    station_id VARCHAR(32) PRIMARY KEY, -- e.g., 'CT-MOH-14'
    range_zone VARCHAR(32),             -- 'CORE', 'BUFFER'
    sub_range VARCHAR(32),              -- 'MOHARLI', 'KOLSA', 'TADOBA'
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    elevation REAL
);

-- Sightings Ledger (The Master Attendance Record)
CREATE TABLE sightings_log (
    sighting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tiger_id VARCHAR(32) REFERENCES tigers(tiger_id),
    station_id VARCHAR(32) REFERENCES camera_stations(station_id),
    image_filename VARCHAR(256),
    sighting_timestamp TIMESTAMP NOT NULL,
    flank_detected VARCHAR(8),
    confidence_score REAL,
    is_verified_by_human BOOLEAN DEFAULT 0,
    sync_status VARCHAR(16) DEFAULT 'LOCAL_ONLY'
);

-- Curated Exemplar Gallery
CREATE TABLE exemplar_gallery (
    exemplar_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tiger_id VARCHAR(32) REFERENCES tigers(tiger_id),
    file_path VARCHAR(256) NOT NULL,
    flank_side VARCHAR(8) NOT NULL,
    modality VARCHAR(8) NOT NULL,       -- 'RGB_DAY', 'IR_NIGHT'
    sharpness_score REAL NOT NULL,
    alignment_score REAL NOT NULL,
    is_primary BOOLEAN DEFAULT 0
);
```

---

## 5. 🔄 Offline-First Delta Sync Protocol

The mobile field app and desktop workstation maintain strict data consistency using a local timestamp synchronization protocol over local camp Wi-Fi or ad-hoc hotspot:

```text
[ Field Mobile Device (SQLite) ]                 [ Desktop Workstation (Master DB) ]
               │                                                 │
               │ ─── 1. Connect Local Wi-Fi / Hotspot ─────────► │
               │                                                 │
               │ ─── 2. Request Sync: Last_Sync_Timestamp ─────► │
               │                                                 │
               │ ◄── 3. Delta Vector Payload (New Tigers/Profiles)
               │        (Updates Mobile SQLite Vector Index)     │
               │                                                 │
               │ ─── 4. Push Local Sighting Rows & Cached Crops ─►
               │        (Appends Field Sightings to Master DB)   │
               │                                                 │
               │ ◄── 5. Sync Confirmation & Checksum Match ───── │
```

---

## 6. 📅 Ten-Day Engineering & Execution Roadmap

### 🗓️ Days 1–2: Data Pipeline & Model Groundwork
* Download the ATRW dataset and split images by identity and flank entity (`Entity = ID + Flank`).
* Build the PyTorch Dataset loader featuring data augmentations: random erasing, simulated IR grayscale conversion, and contrast adjustments.
* Establish baseline verification metrics (mAP, Rank-1, Rank-5 CMC).

### 🗓️ Days 3–4: Part-Based Re-ID Architecture
* Implement the HRNet inference wrapper to predict the 15 keypoints.
* Write the 7-part Axis-Aligned Bounding Box (AABB) cropping module.
* Assemble the PPbM-b architecture: ResNet-50 intermediate RAP pooling (`res3d`) coupled to the Squeeze-and-Excitation attention head.
* Train the network using Batch-Hard Triplet Loss (TriHard).

### 🗓️ Days 5–6: Export & Local Backend Development
* Export models to ONNX and test CPU inference speeds via ONNX Runtime.
* Write the FastAPI service: endpoints for `/batch-cull`, `/reid-match`, and `/exemplar-select`.
* Construct the SQLite schema and seed it with known camera stations and simulated GPS coordinates.

### 🗓️ Days 7–8: Workstation Dashboard & Exemplar Gallery
* Build the desktop frontend interface.
* Implement the batch folder drag-and-drop workflow, human confirmation split-screen, and automatic gallery ranking view.
* Integrate Leaflet/OpenStreetMap to plot tiger movement trajectories and calculate home-range convex polygons.

### 🗓️ Day 9: Ranger Mobile App Construction
* Build the Flutter mobile companion app with on-device camera integration and simulated OTG ingestion.
* Integrate on-device matching with pre-loaded vector weights and hardware GPS coordinate stamping.
* Test the local delta-sync workflow between mobile and desktop.

### 🗓️ Day 10: Pitch Deck, Edge Cases & Live Demo Rehearsal
* Stress-test challenging edge cases: night IR images, partial grass occlusion, and unregistered specimens.
* Finalize the 5-minute presentation deck and prepare live demonstration assets.

---

## 7. 🏆 The Winning 5-Minute Pitch & Live Demo Script

### ⏱️ Minute 0:00 – 1:00: The Problem Statement (Ground Reality)
> *"Reserves like Tadoba have successfully deployed perimeter AI cameras for human-wildlife safety, but inside the 1,700 square kilometers of deep jungle, there is no internet, no 5G, and no power grid.*  
>  
> *Every month, rangers physically collect hundreds of SD cards containing thousands of camera-trap photos. Wildlife biologists spend weeks manually squinting at stripe patterns to compile census reports and track conflict animals.*  
>  
> *We built TigerID: a complete, offline-first biometric re-identification platform that turns raw SD cards into operational conservation intelligence in minutes, with zero cloud dependency."*

---

### ⏱️ Minute 1:00 – 2:30: Live Demonstration (The Desktop Station)
* **The Ingestion:** Drag and drop a folder of 100 images into the desktop dashboard.
* **The Processing:** Watch the system process the batch in real-time. The interface flags that 60 blank/cattle photos were culled, isolates 40 tiger images, and extracts the 15-point skeletons.
* **The Match:** The screen displays a side-by-side verification: a nighttime IR photo matches against Tiger T-04 with **94.2% confidence**.
* **The Exemplar Gallery:** Show the automatically curated gallery for T-04, highlighting how the system updated its profile with the sharpest canonical shots.
* **The Intelligence:** Navigate to the GIS map. Show how the points automatically generated T-04's territorial trajectory across three camera stations.
* **The Natural Language Query:** Type into the console: *"Show all detections of T-04 this week"*, and watch the system instantly filter the records and highlight the route.

---

### ⏱️ Minute 2:30 – 3:30: The Algorithmic Edge (SynoraAI Criterion)
> *"We didn't use generic image classification. We implemented an open-set Pose Part-Based Model (PPbM-b) derived from recent wildlife research.*  
>  
> *Our network uses HRNet to detect 15 skeletal points, extracts 7 localized anatomical regions using Regional Average Pooling at layer res3d, and dynamically weights visible stripe patterns using Squeeze-and-Excitation soft attention. If tall grass blocks the front leg, the attention module down-weights that limb and relies on the flank stripes.*  
>  
> *We optimize using TriHard metric learning, allowing our system to accurately identify uncataloged tigers without requiring network retraining."*

---

### ⏱️ Minute 3:30 – 4:30: The Field Mobile Companion
* Switch to the mobile device in **Airplane Mode**.
* Demonstrate snapping a photo of an animal or plugging in a mock SD card.
* Show the phone extracting coordinates via **hardware GPS satellites**, matching the vector locally against the SQLite database, and flagging that the animal is operating outside its known territory.

---

### ⏱️ Minute 4:30 – 5:00: Deployment & Value Proposition
* **Zero cloud subscription fees**
* **Zero external GPU requirements**
* Data ownership remains **100% with the Maharashtra Forest Department**
* Output matches the exact **SECR formatting** required for the National Tiger Conservation Authority census.