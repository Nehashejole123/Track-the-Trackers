System Architecture: TigerID Core & Field Platform
This document outlines the high-level system topology, data flow, machine learning pipelines, and component boundaries for the TigerID biometric re-identification platform. The system is designed for an offline-first, B2G (Business-to-Government) deployment environment, specifically tailored for forest reserves lacking cellular or internet infrastructure.

1. High-Level System Topology
The architecture is divided into two primary nodes operating independently and syncing via local network protocols:

Biologist Workstation (Base Camp Node): A desktop-bound system running a local React/Vite frontend, a FastAPI backend, and ONNX-accelerated inference. It serves as the master database for processing bulk camera-trap SD cards, visualizing GIS trajectories, and curating the exemplar gallery.

Ranger Field Companion (Mobile Edge Node): An Android application built in Flutter utilizing TFLite for on-device inference. It allows field guards to verify SD cards via OTG or process live camera captures using hardware GNSS/GPS, entirely in airplane mode.

2. Core AI Pipeline & Data Flow
The biometric identification process follows a strict 4-stage pipeline designed to filter noise, handle non-rigid animal postures, and execute open-set metric matching.

Plaintext
[ Raw SD Card / Camera Trap Ingestion ]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: SPECIES CULLING & IMAGE QUALITY ASSESSMENT (IQA)   │
│ - Model: YOLOv8 / MegaDetector-v5                           │
│ - Logic: Discard empty frames, foliage, non-target fauna    │
│ - Metric: Laplacian Variance (Blur) + Exposure Histograms   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Filtered High-Quality Tiger Crops)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: BIOMETRIC PARSING & SKELETON POSE ALIGNMENT        │
│ - Orientation Net: Left Flank vs. Right Flank Classifier    │
│ - Pose Model: HRNet (15 Anatomical Keypoints)               │
│ - Decomposition: 7-Part Axis-Aligned Bounding Boxes (AABB)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: DEEP METRIC RE-ID MATCHING ENGINE                  │
│ - Feature Extractor: ResNet-50 Intermediate RAP (res3d)     │
│ - Attention Head: Squeeze-and-Excitation (PPbM-b)           │
│ - Inference: 512-D L2-Normalized Embedding Vector           │
│ - Search: Local FAISS Index / Cosine Similarity Engine      │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
[ Match Found: Cosine Sim ≥ 0.75 ]    [ Unregistered: Cosine Sim < 0.75 ]
  - Tiger ID Assigned                   - Flagged as "UNKNOWN_NEW"
  - Sighting Attendance Logged          - Routed to Human Verification
            │                                     │
            └──────────────────┬──────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: EXEMPLAR SELECTION & GALLERY CURATION              │
│ - Rank images by: Composite Score = Sharpness × Alignment   │
│ - Keep Top-5 Exemplars in local storage                     │
│ - Write spatial log to SQLite / SECR-compatible CSV         │
└─────────────────────────────────────────────────────────────┘
3. Component Boundaries & Tech Stack
A. Desktop Workstation (Master Node)
Frontend: React + Vite + Tailwind CSS.

Backend API: FastAPI (Python).

Machine Learning Execution: ONNX Runtime (CPU optimized for standard government hardware).

Vector Search Engine: FAISS (Facebook AI Similarity Search) index flat file (.index).

Relational Database: SQLite (tatr_master.db).

GIS Mapping: Leaflet / Mapbox GL (using pre-cached offline map tiles).

B. Mobile Field Companion (Edge Node)
Frontend & Logic: Flutter (Android APK target).

Machine Learning Execution: TensorFlow Lite (TFLite) or ONNX Runtime Mobile.

Local Storage: SQLite via sqflite plugin.

Hardware Integration: USB-C OTG storage access and Hardware GNSS (geolocator) for coordinate stamping without cellular data.

4. Ingestion & Synchronization Protocols
SD Card Batch Ingestion Flow (Desktop)
Mount: User connects SD card containing /DCIM/ burst captures.

Map: System extracts Camera Station ID from the folder nomenclature (e.g., CT_MOHARLI_14) and joins it with pre-indexed GPS coordinates.

Process: Images are batched through the 4-Stage AI Pipeline.

Confirm: Edge-case and "Unknown" matches are queued in the UI for a Human-in-the-Loop visual confirmation.

Export: Verified data is committed to the database and exported to SECR-compliant CSV formats.

Field-to-Base Delta Sync Protocol (Mobile)
To maintain data consistency across disconnected field devices and the master desktop, the system utilizes a local timestamp-based delta sync over ad-hoc Wi-Fi networks at base camps.

Plaintext
[ Mobile Field Device (SQLite) ]                 [ Desktop Workstation (Master DB) ]
               │                                                 │
               │ ─── 1. Connect to Local Camp Wi-Fi ───────────► │
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