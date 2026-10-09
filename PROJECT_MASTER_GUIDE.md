# 🐅 TigerID Biometric Platform
> **Master Blueprint & Architecture Specification**  
> *Offline-First Biometric Re-Identification & Intelligence Platform for Tadoba Andhari Tiger Reserve (TATR)*

---

## 📖 Overview

This document serves as the master blueprint for the **TigerID** biometric re-identification platform. It translates the operational requirements of the **Tadoba Andhari Tiger Reserve (TATR)** and the algorithmic criteria of **SynoraAI Labs** into an end-to-end software system. It explains the system architecture, machine learning workflows, user interfaces, and file roles without unnecessary mathematical jargon.

---

## 1. Executive Summary & Ground Reality

### 🚨 The Core Problem
**Tadoba Andhari Tiger Reserve (TATR)** covers over **1,700 square kilometers**. Inside the core forest, there is:
* **Zero cellular network**
* **No Wi-Fi connectivity**
* **No electrical grid**

Wildlife monitoring relies on hundreds of motion-activated, battery-powered camera traps strapped to trees. Every 15 to 30 days, forest guards physically walk through the jungle, swap out SD cards, and bring tens of thousands of raw photos back to headquarters.

> [!WARNING]
> **The Operational Bottleneck:**  
> Up to **80%** of these photos contain no animals at all (triggered by swaying grass, shadows, or wind), while others capture non-target animals like deer and cattle. Biologists currently spend weeks manually opening folders, sorting photos, and squinting at stripe patterns to answer:  
> *"Is this Tiger T-54, or is it a new intruder?"*

### 💡 What We Are Building
**TigerID** is an offline-first biometric intelligence platform consisting of two synchronized tools:

1. **🖥️ The Biologist Desktop Workstation:**  
   A local web dashboard for base-camp computers that ingests an entire camera-trap SD card with one drag-and-drop, filters out non-target images, biometrically matches tiger stripes against known profiles, updates sighting attendance logs, curates image galleries, and maps territorial movement corridors.

2. **📱 The Ranger Mobile Field Companion:**  
   A lightweight offline Android app that allows field guards to inspect SD cards on the spot using a USB-C OTG card reader or identify tigers during field encounters using satellite GPS, without needing internet connectivity.

---

## 2. The Two Interfaces: Operational Use Cases

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   INTERFACE 1: BIOLOGIST WORKSTATION                   │
│         (React + Vite + Tailwind + FastAPI on Base Camp PC)            │
├────────────────────────────────────────────────────────────────────────┤
│ • Bulk SD Ingestion: Processes 500+ photos in seconds locally.         │
│ • Noise Culling: Drops blank images, swaying grass, cattle, and deer.  │
│ • Stripe Matching: Matches against local SQLite vector database.       │
│ • One-Click Verification: Side-by-side human confirmation view.        │
│ • Master Exemplar Gallery: Auto-updates Top-5 cleanest tiger photos.   │
│ • GIS Movement Map: Draws movement paths across camera stations.       │
│ • Census Exporter: Exports official SECR-compliant attendance sheets.  │
│ • Local RAG Assistant: Answers plain-English sighting questions.       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                         Local Wi-Fi Delta Sync
                      (No Cloud / No Server Costs)
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                   INTERFACE 2: RANGER FIELD COMPANION                  │
│               (Flutter + SQLite + TFLite on Android)                   │
├────────────────────────────────────────────────────────────────────────┤
│ • OTG Card Reader Mode: Plug camera-trap SD card directly into phone.  │
│ • Live Encounter Capture: Instant identification in Airplane Mode.     │
│ • Hardware GNSS Stamping: Records satellite GPS coordinates offline.   │
│ • Boundary Alerts: Warns if a core tiger wanders into a buffer zone.   │
│ • Local Caching: Saves field sightings to sync back at base camp.      │
└────────────────────────────────────────────────────────────────────────┘
```

### 🖥️ Scenario A: The Base-Camp Workstation (Desktop Web)

* **Who uses it:** Wildlife biologists, senior forest officers, research fellows.
* **Where:** Range forest offices (Moharli, Kolsa, Tadoba).
* **Hardware:** Standard-issue office desktop or laptop (Intel Core i5/i7, 16 GB RAM, no dedicated GPU required).
* **The Workflow:** The officer mounts an SD card folder (e.g., `CT_KOLSA_08`). The app scans the images locally, discards blank frames, crops tigers, matches stripes, updates the database, and flags unresolved photos for quick human approval.

### 📱 Scenario B: The Field Companion (Mobile App)

* **Who uses it:** Forest beat guards, Rapid Response Units (RRU), anti-poaching squads.
* **Where:** Deep jungle beats, buffer boundaries, village fringes.
* **Hardware:** Standard Android smartphone running in Airplane Mode.
* **The Workflow:**
  * **OTG Inspection:** A guard connects a compact USB-C card reader to their phone, inserts the camera's SD card, and scans recent captures right at the tree to confirm recent activity before leaving the beat.
  * **Conflict / Carcass Encounter:** When a tiger is involved in a livestock kill or tranquilized near a village, the guard captures a photo. The on-device model provides an immediate match confirmation, allowing the team to verify if the animal has a conflict history.

---

## 3. How the AI Models Work (Simplified Architecture)

To maintain stability and low latency, the system utilizes two focused models supported by algorithmic validation checks, rather than multiple complex sub-networks.

```text
Incoming Photo (Daytime Color or Nighttime Infrared)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   MODEL 1: TIGER DETECTOR        │
        │   (YOLOv8 Animal Localizer)      │
        └────────────────┬─────────────────┘
                         │
           Is there a tiger in the image?
           ├── NO  ──► Discard (Empty trees, cattle, deer)
           └── YES ──► Crop Tiger Bounding Box
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   QUALITY & ORIENTATION CHECK    │
        │   (Sharpness & Flank Classifier) │
        └────────────────┬─────────────────┘
                         │
           • Blur Check: Is the image too blurry? (Laplacian test)
           • Flank Check: Left Side (0) vs. Right Side (1)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   MODEL 2: BIOMETRIC EXTRACTOR   │
        │   (Pose-Aware ResNet-50 / PPbM)  │
        └────────────────┬─────────────────┘
                         │
     Converts stripe patterns into a 512-number vector
     (A mathematical fingerprint of the tiger's coat)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   LOCAL VECTOR SIMILARITY ENGINE │
        │   (Cosine Similarity Search)     │
        └────────────────┬─────────────────┘
                         │
        Compares query vector against SQLite database:
        ├── Score ≥ 75%: Match Confirmed (e.g., "T-54 Matkasur")
        ├── Score 65–75%: Uncertain (Needs Biologist Confirmation)
        └── Score < 65%: Brand New Tiger (Flagged as Unregistered)
```

### 🔍 Model 1: The Gatekeeper (YOLOv8)
* **What it does:** Scans the full high-resolution image to locate animal shapes.
* **Why it matters:** It filters out empty images caused by moving vegetation and non-target animals, reducing thousands of raw photos to a manageable set of cropped tiger images.

### 📐 Pre-Extraction Checks: Sharpness & Flank Routing
* **Sharpness Gate:** Computes edge variation (Laplacian variance) on the crop. If an image is severely blurred from fast running motion, it is routed to a manual review queue to prevent degraded data from affecting the database.
* **Flank Classifier:** A lightweight classifier sorts the crop into **Left Flank** or **Right Flank**. Because a tiger's stripes are completely asymmetrical (the left side does not match the right side), the left and right profiles are indexed as separate biometric entities.

### 🧬 Model 2: The Biometric Fingerprint Extractor (ResNet-50 / PPbM)
* **What it does:** Converts the cropped tiger image into an array of 512 numbers (a vector embedding).
* **The Concept:** Think of this array as a coordinate in 512-dimensional space. Photos of the same tiger map close together, while photos of different tigers map far apart.
* **Cosine Similarity:** To determine identity, the software calculates the cosine similarity (directional overlap) between the query array and database arrays. A score of $0.94$ indicates a 94% match.

---

## 4. Solving Real-World Field Challenges

| Challenge | Why Basic AI Fails | How TigerID Solves It |
| :--- | :--- | :--- |
| **🌿 Tall Grass & Tree Occlusion** | Standard models average the entire bounding box, causing the network to memorize grass patterns instead of fur. | **Soft-Attention Mechanism (PPbM):** The network extracts features across distinct anatomical parts (trunk, legs, thighs). If grass covers a front leg, the attention module lowers that limb's weight and relies on the visible flank stripes. |
| **🌓 Day vs. Night (IR Cameras)** | Daytime photos are color (RGB); night photos are black-and-white infrared (IR). Basic models treat them as different animals. | **Cross-Modality Augmentation:** Training pipelines convert color training samples into high-contrast grayscale and noisy textures, training the network to focus on stripe geometry rather than color. |
| **🐾 Severe Body Angles & Walking Poses** | A turning or crouching tiger deforms its stripe patterns, altering distances between markings. | **Skeletal Alignment (HRNet Keypoints):** Keypoint estimation tracks 15 joint positions (shoulders, hips, knees), normalizing physical postures into standardized coordinate frames. |
| **❓ Unregistered Intruders & Cubs** | Closed classification models only predict known classes, forcing new tigers into incorrect existing IDs. | **Open-Set Deep Metric Learning:** The model is trained using TriHard triplet loss to extract distance metrics rather than static class labels. Any vector with a similarity score below 65% is safely categorized as a new individual. |

---

## 5. Dataset Strategy: Training Once to Scale Universally

> [!NOTE]
> **Core Technical Question:**  
> *If we train our model on the 92 tigers in the ATRW dataset, will it work on the 115+ tigers in Tadoba without breaking?*  
> 
> **Yes**, because the model learns **pattern recognition** rather than individual identities. Standard classifiers memorize specific faces. Metric learning models learn **stripe grammar**: how stripes branch, curve, narrow, and terminate under varying illumination. Once the network learns to differentiate stripe patterns across the 92 tigers in the ATRW dataset, it can separate novel stripe patterns across hundreds of unseen animals without weight retraining.

### 🎯 The Hackathon Demonstration Strategy
* **The Model Engine:** Trained using the public **ATRW** (*Amur Tiger Re-identification in the Wild*) benchmark.
* **Pre-Seeded Profiles:** Pre-populate the local database with reference profiles using historical photos of well-known Tadoba tigers (such as **Matkasur T-54**, **Maya T-12**, and **Choti Madhu T-114**).
* **The Live Test:**
  * **Test 1 (Known Animal):** Ingest an angled or night-vision photo of Matkasur. The system matches his profile with **$>90\%$ confidence** and logs his sighting attendance.
  * **Test 2 (Novel Animal):** Ingest a photo of an un-enrolled tiger. The system returns a maximum score **below $65\%$**, flags it as an *"Unregistered Specimen"*, and allows one-click enrollment into the database.

---

## 6. Innovative Features Designed for Conservation

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   KEY INNOVATION MODULES OVERVIEW                      │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Master Exemplar Gallery:                                            │
│    Auto-ranks photos using Laplacian sharpness to keep the cleanest    │
│    5 canonical images per tiger for visual validation.                 │
├────────────────────────────────────────────────────────────────────────┤
│ 2. GIS Trajectory & Spatiotemporal Movement:                           │
│    Connects chronological camera-station logs into interactive vectors │
│    to calculate home ranges and highlight territorial shifts.          │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Automated SECR Census Sheet Generation:                             │
│    One-click export of official National Tiger Conservation Authority  │
│    tabular reports (SightingID, IndividualID, DetectorID, Date, Time). │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Local Natural Language Query Agent (RAG):                           │
│    Converts plain-English questions into local database queries to     │
│    return sighting timelines and map plots without internet access.    │
├────────────────────────────────────────────────────────────────────────┤
│ 5. Offline-First Delta Synchronization:                                │
│    Enables mobile devices to upload sightings and fetch new vectors    │
│    over local base-camp Wi-Fi with zero external cloud dependencies.   │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. 🖼️ The Automated Exemplar Gallery
Instead of cluttering storage with repetitive, blurry burst captures, the system scores every confirmed detection using edge sharpness and pose completeness. The cleanest five images (daytime flanks, nighttime infrared, full body) are automatically cataloged into `tigers/{tiger_id}/gallery/`. When a clearer photo is captured, the gallery updates its primary reference image automatically.

### 2. 🗺️ GIS Trajectory & Movement Corridors
Every camera trap is tied to a **Station ID** with recorded latitude and longitude coordinates. When a tiger is identified across different stations over time, the system renders an interactive route map using Leaflet.
* **Territory Calculation:** Generates Minimum Convex Polygons (MCP) to compute active home-range territories in square kilometers.
* **Intrusion Alerts:** If an uncataloged sub-adult tiger is detected inside a resident dominant male's territory, the system flags the overlap to help managers anticipate dispersal toward village boundaries.

### 3. 📊 SECR National Tiger Census Exporter
Every four years, India conducts the All India Tiger Estimation using Spatially Explicit Capture-Recapture (SECR) models. The desktop workstation includes a one-click exporter that converts verified sightings into NTCA-standard format:

$$\text{Columns: } [\text{SightingID}, \text{IndividualID}, \text{Sex}, \text{DetectorID}, \text{Latitude}, \text{Longitude}, \text{Date}, \text{Time}]$$

### 4. 🤖 Local RAG Query Assistant
An embedded offline language tool running locally via lightweight query parsers translates conversational queries into structured SQL lookups:
* *"When was T-54 last detected in the Kolsa range?"*
* *"Show all tiger sightings logged after 6:00 PM near Moharli."*
* *"Which individuals appeared at Camera Station 12 this week?"*

The system executes the query on `tatr_local.db` and returns structured results with corresponding map markers.

### 5. 🔄 Offline-First Delta Synchronization
Field phones and desktop workstations synchronize peer-to-peer over local base-camp Wi-Fi. The phone requests updates newer than its last sync timestamp, receives lightweight vector payloads for newly registered tigers, and pushes queued field sightings to the master desktop database.

---

## 7. Structure of the Generated Documentation Files

The technical implementation is split across four functional specifications:

| Document | Focus Area | Description & Operational Value |
| :--- | :--- | :--- |
| **`ARCHITECTURE.md`** | System Topology & Data Flow | **What it covers:** End-to-end component layout, network boundaries, the four-stage AI processing pipeline, and peer-to-peer synchronization flows.<br>**Why it matters:** Outlines how the base-camp workstation and mobile field companion operate together without cloud dependencies. |
| **`MODEL_PIPELINE.md`** | Machine Learning Specifications | **What it covers:** Input/output tensor shapes, YOLOv8 detector specifications, Laplacian blur thresholds, ResNet-50 feature extraction, and cosine similarity cutoffs.<br>**Why it matters:** Establishes model contracts, ensuring inference runs cleanly within memory limits on local hardware. |
| **`DATABASE_SCHEMA.md`** | Relational Data & Vector Storage | **What it covers:** SQLite DDL scripts for `tigers`, `biometric_profiles`, `camera_stations`, `sightings_log`, and `exemplar_gallery`, including vector serialization logic.<br>**Why it matters:** Defines unified storage for metadata and 512-D float vectors across both Python and Flutter platforms. |
| **`API_SPEC.md`** | FastAPI Endpoints | **What it covers:** Endpoints for `/ingest/batch`, `/reid/match`, `/sightings/verify`, `/gallery/curate`, and `/sightings/trajectory`.<br>**Why it matters:** Connects the React/Vite user interface to the Python processing engine through standardized JSON and multipart endpoints. |

---

## 8. Complete Technology Stack

| Layer / Component | Technology Stack | Key Role / Deployment Details |
| :--- | :--- | :--- |
| **Biologist Workstation** | React (Vite), Tailwind CSS, Lucide Icons | Base Camp Browser UI |
| **Workstation Backend** | Python 3.10+, FastAPI, Uvicorn | Local REST Service & Pipeline Host |
| **Inference Runtime** | ONNX Runtime / OpenVINO (CPU Optimized) | High-speed local CPU inference |
| **Ranger Field App** | Flutter SDK (Dart, Android APK) | Mobile offline application |
| **Mobile Inference** | TensorFlow Lite (TFLite FP16) | On-device lightweight model execution |
| **Relational Storage** | SQLite 3 (WAL Mode Enabled) | Local atomic database storage |
| **Vector Math** | NumPy (Dot-Product Cosine Similarity) / FAISS | High-speed vector metric comparison |
| **Mapping & GIS** | Leaflet.js / React-Leaflet (Offline Tiles) | Interactive GIS corridor & territory mapping |
| **Computer Vision** | OpenCV, Albumentations, PyTorch (Training) | Image preprocessing, cropping, and training |

---

## 9. Hackathon Live Demonstration Script

To present this system effectively to both the **Tadoba Forest Department officials** and the **SynoraAI technical judges**, structure the live demonstration in five clear stages:

### ⏱️ Minute 1: The Operational Challenge
**Address the panel directly:**
> *"Tadoba monitors hundreds of square kilometers using offline camera traps strapped to trees. Every month, guards collect hundreds of SD cards packed with thousands of unorganized images. Biologists spend weeks manually sorting through empty shots and identifying stripes. We built TigerID to turn raw SD cards into verified population intelligence within minutes, running 100% locally on standard office computers."*

---

### ⏱️ Minute 2: Bulk SD Ingestion & AI Processing
**Demonstrate the desktop dashboard live:**
1. **Drag and drop** a folder of 100 test camera-trap photos into the web interface.
2. **Show the pipeline** culling blank shots and cattle images in real time, leaving only clean tiger crops.
3. **Show the biometric match:** an infrared nighttime photo matches against known resident Tiger **T-54 (Matkasur)** with **94.2% confidence**.
4. **Open the human verification split-screen** and click **"Approve & Log"**, updating the official sighting attendance record.

---

### ⏱️ Minute 3: The Algorithmic Architecture
**Address the technical evaluation team:**
1. **Explain** that the system avoids closed-set classification in favor of **Open-Set Metric Learning**.
2. **Highlight** how the Pose Part-Based Model (**PPbM**) and soft-attention mechanisms allow the model to down-weight limbs obscured by grass and match visible flank stripes reliably.
3. **Run an un-enrolled test photo:** show the system scoring it below **$65\%$**, flagging it as an *"Unregistered Tiger"*, and registering it with a single click without needing model retraining.

---

### ⏱️ Minute 4: The Ranger Mobile Companion
**Switch to the Android smartphone in Airplane Mode:**
1. **Connect** a USB-C OTG card reader containing camera-trap photos to the phone.
2. **Run an offline scan** to identify the tiger locally.
3. **Show the app reading satellite GPS coordinates** directly from hardware without cellular reception and saving the sighting to the local database.

---

### ⏱️ Minute 5: Conservation Impact & Value
**Present the analytical outputs:**
1. **Display the GIS Trajectory Map**, showing Matkasur's movement vectors across three camera stations.
2. **Click "Export SECR Census"** to generate an NTCA-compliant attendance spreadsheet instantly.
3. **Conclude by emphasizing the deployment model:** zero recurring cloud costs, zero proprietary hardware requirements, and complete data privacy for the reserve management.