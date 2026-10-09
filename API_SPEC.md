API_SPEC.md
This document defines the local FastAPI contracts required to connect the React/Vite workstation frontend with the Python ONNX/SQLite backend.

Agent Instruction for Implementation: Use FastAPI with Pydantic for schema validation. Since this is a locally hosted B2G (Business-to-Government) application, CORS should be configured to allow localhost origins, and file upload limits should be increased to handle large batch processing. Heavy inference tasks must not block the main thread; utilize FastAPI's BackgroundTasks or asyncio.to_thread.

1. Global API Configuration
Base URL: http://localhost:8000/api/v1

Data Serialization: JSON for standard payloads; multipart/form-data for direct image uploads.

Error Handling: Standard HTTP status codes with a JSON body {"error": "string", "detail": "string"}.

2. Endpoint Definitions
A. Batch SD Card Ingestion & Filtering
POST /ingest/batch

Description:
Triggers the Stage 1 pipeline (YOLOv8 Detection & Laplacian Blur Check). Because this app runs locally on the forest office PC, passing a local absolute directory path is the most memory-efficient ingestion method, avoiding HTTP payload limits.

Request Body (application/json):

JSON
{
  "source_directory": "D:/DCIM/CT_MOHARLI_14",
  "station_id": "CT-MOH-14",
  "run_reid_pipeline": true 
}
(Agent Note: If run_reid_pipeline is true, automatically pass valid crops to the Model 2 extractor).

Response (200 OK):

JSON
{
  "status": "COMPLETED",
  "metrics": {
    "total_images_scanned": 520,
    "blanks_and_noise_discarded": 420,
    "blur_rejected": 18,
    "valid_tiger_crops_extracted": 82
  },
  "processing_time_ms": 14500
}
B. Biometric Re-ID Matching
POST /reid/match

Description:
Accepts a cropped image, runs the HRNet orientation/pose check, extracts the 512-D PPbM vector via ONNX ResNet-50, and performs a cosine similarity search against biometric_profiles in the SQLite database.

Request Body (multipart/form-data):

image: The raw cropped image file (JPEG/PNG).

station_id (Optional): String indicating capture location.

timestamp (Optional): ISO 8601 string from EXIF data.

Response (200 OK):

JSON
{
  "inference_ms": 112,
  "flank_detected": "LEFT",
  "match_status": "MATCH_CONFIRMED", 
  "top_candidates": [
    {
      "tiger_id": "T-04",
      "common_name": "Matkasur",
      "confidence_score": 0.942,
      "reference_image_path": "/data/catalog/T-04/left_primary.jpg"
    },
    {
      "tiger_id": "T-12",
      "common_name": "Maya",
      "confidence_score": 0.410,
      "reference_image_path": "/data/catalog/T-12/left_primary.jpg"
    }
  ],
  "pending_sighting_id": 1045
}
(Agent Note: match_status enum: MATCH_CONFIRMED [>0.75], MATCH_UNCERTAIN [0.65-0.75], UNKNOWN_NEW [<0.65].)

C. Human-in-the-Loop Verification
POST /sightings/verify/{sighting_id}

Description:
Called when a biologist manually confirms a match or registers an unknown tiger from the dashboard UI. Updates the verified_flag in the database.

Request Body (application/json):

JSON
{
  "action": "CONFIRM_EXISTING", 
  "tiger_id": "T-04",
  "update_gallery": true
}
(Agent Note: action enum can be CONFIRM_EXISTING, REGISTER_NEW, or REJECT_FALSE_POSITIVE)

Response (200 OK):

JSON
{
  "success": true,
  "sighting_id": 1045,
  "tiger_id": "T-04",
  "message": "Sighting logged and attendance updated."
}
D. Automated Exemplar Gallery Curation
POST /gallery/curate

Description:
Evaluates the sharpness (Laplacian variance) and pose alignment score of a new sighting against the existing Top-5 images for that individual tiger. If the new image is better, it replaces the lowest-scoring image in the exemplar_gallery table.

Request Body (application/json):

JSON
{
  "tiger_id": "T-04",
  "sighting_id": 1045
}
Response (200 OK):

JSON
{
  "gallery_updated": true,
  "replaced_exemplar_id": 42,
  "new_sharpness_score": 0.94,
  "current_primary_image": "/data/catalog/T-04/left_best_1045.jpg"
}
E. Trajectory & Territory Analytics (GIS Mapping)
GET /sightings/trajectory/{tiger_id}

Description:
Returns ordered geographical coordinates (GeoJSON format) for a specific tiger to plot movement corridors and compute Minimum Convex Polygons (MCP) on the React frontend via Leaflet.

Query Parameters:

start_date (Optional): YYYY-MM-DD

end_date (Optional): YYYY-MM-DD

Response (200 OK):

JSON
{
  "tiger_id": "T-04",
  "total_sightings": 64,
  "trajectory": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [79.3218, 20.2452] 
        },
        "properties": {
          "station_id": "CT-MOH-01",
          "timestamp": "2026-10-01T18:45:00Z",
          "flank": "LEFT"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [79.2941, 20.2189]
        },
        "properties": {
          "station_id": "CT-MOH-02",
          "timestamp": "2026-10-03T02:14:00Z",
          "flank": "RIGHT"
        }
      }
    ]
  }
}
(Agent Note: Coordinates in GeoJSON are strictly [longitude, latitude].)