How to Handle Vector Embeddings in SQLiteBecause standard SQLite does not have a native vector datatype, store the 512-dimensional vector as a binary BLOB:In Python (Backend):Pythonimport numpy as np

# Writing to DB: Convert float32 array to raw 2048 bytes
embedding = np.random.randn(512).astype(np.float32)
blob_data = embedding.tobytes()

# Reading from DB: Unpack 2048 bytes back to numpy array
restored_vector = np.frombuffer(row["embedding_blob"], dtype=np.float32)
Performance: Calculating cosine similarity across 200 tiger vectors using NumPy in Python takes under $1\text{ ms}$. You do not need complex vector database servers for this dataset size.DATABASE_SCHEMA.mdThis document defines the relational database architecture, table definitions, indexing strategies, and binary serialization standards for tatr_local.db.Plaintext       ┌──────────────┐
       │    tigers    │
       └──┬────────┬──┘
          │ 1      │ 1
          │        │
        N │      N │
┌─────────▼──────────┐ ┌─────────▼──────────┐
│ biometric_profiles │ │  exemplar_gallery  │
└────────────────────┘ └────────────────────┘

       ┌─────────────────┐
       │ camera_stations │
       └────────┬────────┘
                │ 1
                │
                │ N
       ┌────────▼────────┐
       │  sightings_log  │
       └─────────────────┘
1. Engine Configuration & Runtime PragmasTo ensure fast read/write operations and strict data integrity under concurrent folder processing, the SQLite connection must execute the following pragmas immediately upon initialization:SQL-- Enforce foreign key constraints
PRAGMA foreign_keys = ON;

-- Enable Write-Ahead Logging for high-concurrency read/write operations
PRAGMA journal_mode = WAL;

-- Optimize disk synchronization for speed while maintaining safety
PRAGMA synchronous = NORMAL;

-- Store temporary tables and indices in memory
PRAGMA temp_store = MEMORY;

-- Set cache size to 64MB for instant table scans
PRAGMA cache_size = -64000;
2. Table Definitions (DDL)SQL-- ============================================================================
-- 1. TIGERS (Master Entity Table)
-- Represents unique physical tigers cataloged in the reserve.
-- ============================================================================
CREATE TABLE IF NOT EXISTS tigers (
    tiger_id VARCHAR(32) PRIMARY KEY,              -- Reserve ID (e.g., 'T-04', 'T-54', 'UNKNOWN_2026_01')
    common_name VARCHAR(64),                       -- Field name (e.g., 'Matkasur', 'Maya', 'Choti Madhu')
    sex VARCHAR(8) CHECK(sex IN ('MALE', 'FEMALE', 'UNKNOWN')) DEFAULT 'UNKNOWN',
    first_recorded_date DATE NOT NULL,             -- Date first captured by camera trap
    status VARCHAR(16) CHECK(status IN ('ACTIVE', 'DISPLACED', 'MISSING', 'DECEASED')) DEFAULT 'ACTIVE',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 2. BIOMETRIC_PROFILES (Feature Embedding Repository)
-- Stores 512-dimensional stripe vectors separated by flank orientation.
-- ============================================================================
CREATE TABLE IF NOT EXISTS biometric_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tiger_id VARCHAR(32) NOT NULL,
    flank_side VARCHAR(8) NOT NULL CHECK(flank_side IN ('LEFT', 'RIGHT')),
    embedding_blob BLOB NOT NULL,                  -- 512 Float32 values = exactly 2048 bytes
    source_image_name VARCHAR(256),                -- Reference image used to derive embedding
    quality_score REAL CHECK(quality_score >= 0.0 AND quality_score <= 1.0),
    is_canonical BOOLEAN DEFAULT 0,                -- 1 if selected as primary match anchor
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tiger_id) REFERENCES tigers(tiger_id) ON DELETE CASCADE
);

-- ============================================================================
-- 3. CAMERA_STATIONS (Spatial Grid & Sensor Deployments)
-- Maps static camera-trap installations to spatial zones and geographic coordinates.
-- ============================================================================
CREATE TABLE IF NOT EXISTS camera_stations (
    station_id VARCHAR(32) PRIMARY KEY,            -- Identifier matching folder name (e.g., 'CT-MOH-14')
    range_zone VARCHAR(16) NOT NULL CHECK(range_zone IN ('CORE', 'BUFFER')),
    sub_range VARCHAR(32) NOT NULL,                -- Administrative beat (e.g., 'Moharli', 'Kolsa', 'Tadoba')
    latitude REAL NOT NULL CHECK(latitude BETWEEN -90.0 AND 90.0),
    longitude REAL NOT NULL CHECK(longitude BETWEEN -180.0 AND 180.0),
    installation_date DATE,
    status VARCHAR(16) CHECK(status IN ('OPERATIONAL', 'MAINTENANCE', 'DECOMMISSIONED')) DEFAULT 'OPERATIONAL'
);

-- ============================================================================
-- 4. SIGHTINGS_LOG (Master Sighting Attendance Ledger)
-- High-volume ledger capturing every confirmed or pending tiger encounter.
-- ============================================================================
CREATE TABLE IF NOT EXISTS sightings_log (
    sighting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tiger_id VARCHAR(32),                          -- Nullable until human-in-the-loop assigns ID
    station_id VARCHAR(32) NOT NULL,
    image_filename VARCHAR(256) NOT NULL,
    sighting_timestamp TIMESTAMP NOT NULL,         -- Extracted from image EXIF metadata
    flank_detected VARCHAR(8) CHECK(flank_detected IN ('LEFT', 'RIGHT', 'UNKNOWN')),
    confidence_score REAL CHECK(confidence_score >= 0.0 AND confidence_score <= 1.0),
    verified_flag BOOLEAN DEFAULT 0,               -- 0 = Algorithmic suggestion; 1 = Approved by Biologist
    sync_status VARCHAR(16) CHECK(sync_status IN ('LOCAL', 'SYNCED_DESKTOP', 'PENDING_FIELD')) DEFAULT 'LOCAL',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tiger_id) REFERENCES tigers(tiger_id) ON DELETE SET NULL,
    FOREIGN KEY (station_id) REFERENCES camera_stations(station_id) ON DELETE RESTRICT
);

-- ============================================================================
-- 5. EXEMPLAR_GALLERY (Curated Visual Profile Repository)
-- Top reference images per individual, optimized for display and human verification.
-- ============================================================================
CREATE TABLE IF NOT EXISTS exemplar_gallery (
    exemplar_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tiger_id VARCHAR(32) NOT NULL,
    image_path VARCHAR(512) NOT NULL,              -- Relative file path in storage
    flank_side VARCHAR(8) NOT NULL CHECK(flank_side IN ('LEFT', 'RIGHT')),
    modality VARCHAR(16) NOT NULL CHECK(modality IN ('RGB_DAY', 'IR_NIGHT')),
    sharpness_score REAL NOT NULL,                 -- Laplacian variance score
    is_primary BOOLEAN DEFAULT 0,                  -- Primary avatar displayed on UI
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tiger_id) REFERENCES tigers(tiger_id) ON DELETE CASCADE
);
3. Performance IndicesSQL-- Rapid indexing for candidate profile retrieval by flank side
CREATE INDEX IF NOT EXISTS idx_biometric_lookup 
ON biometric_profiles (flank_side, tiger_id);

-- Spatial and temporal filters for map tracking and SECR census exports
CREATE INDEX IF NOT EXISTS idx_sightings_time_station 
ON sightings_log (sighting_timestamp DESC, station_id);

CREATE INDEX IF NOT EXISTS idx_sightings_tiger_time 
ON sightings_log (tiger_id, sighting_timestamp ASC);

-- Rapid lookup for human verification queue
CREATE INDEX IF NOT EXISTS idx_sightings_unverified 
ON sightings_log (verified_flag) WHERE verified_flag = 0;

-- Gallery query index
CREATE INDEX IF NOT EXISTS idx_exemplar_tiger 
ON exemplar_gallery (tiger_id, flank_side);
4. Binary Serialization SpecificationsEmbedding Storage FormatEach row in biometric_profiles.embedding_blob stores a 512-dimensional feature vector:Precision: IEEE 754 Single-Precision Floating Point (float32, 4 bytes).Length: $512 \times 4 = 2,048\text{ bytes}$ fixed size.Byte Order: Little-Endian standard (<f4).Serialization ImplementationPython (Workstation Backend)Pythonimport numpy as np


def serialize_vector(vector: np.ndarray) -> bytes:
    """Validates and converts a 512-D float32 numpy array into raw bytes."""
    if vector.shape != (512,) or vector.dtype != np.float32:
        vector = np.ascontiguousarray(vector, dtype=np.float32)
    return vector.tobytes()


def deserialize_vector(blob: bytes) -> np.ndarray:
    """Reconstructs 512-D numpy vector from SQLite binary blob."""
    return np.frombuffer(blob, dtype=np.float32)
Dart (Flutter Mobile Field Companion)Dartimport 'dart:typed_data';

// Packing Float32List to Uint8List for SQLite BLOB storage
Uint8List vectorToBlob(List<double> vector) {
  final floatList = Float32List.fromList(vector);
  return floatList.buffer.asUint8List();
}

// Unpacking SQLite BLOB to Float32List
Float32List blobToVector(Uint8List blob) {
  return blob.buffer.asFloat32List();
}
5. Seed Data & SECR Census ExportSample Seed Data (Tadoba Demonstrations)SQL-- Seed Camera Stations
INSERT OR IGNORE INTO camera_stations (station_id, range_zone, sub_range, latitude, longitude) VALUES
('CT-MOH-01', 'CORE', 'Moharli', 20.2452, 79.3218),
('CT-MOH-02', 'BUFFER', 'Moharli', 20.2189, 79.2941),
('CT-KOL-01', 'CORE', 'Kolsa', 20.1512, 79.4312),
('CT-TAD-01', 'CORE', 'Tadoba', 20.2831, 79.3498);

-- Seed Target Individuals
INSERT OR IGNORE INTO tigers (tiger_id, common_name, sex, first_recorded_date, status) VALUES
('T-54', 'Matkasur', 'MALE', '2019-03-12', 'ACTIVE'),
('T-12', 'Maya', 'FEMALE', '2014-06-21', 'ACTIVE'),
('T-114', 'Choti Madhu', 'FEMALE', '2021-11-04', 'ACTIVE');
Standard SECR Export QueryThe National Tiger Conservation Authority (NTCA) requires data in Spatially Explicit Capture-Recapture formats. The following SQL query produces the exact tabular export required by conservation biologists:SQLSELECT 
    s.sighting_id AS SightingID,
    s.tiger_id AS IndividualID,
    t.sex AS Sex,
    s.station_id AS DetectorID,
    c.latitude AS Latitude,
    c.longitude AS Longitude,
    strftime('%Y-%m-%d', s.sighting_timestamp) AS Date,
    strftime('%H:%M:%S', s.sighting_timestamp) AS Time,
    s.flank_detected AS Flank,
    s.confidence_score AS MatchConfidence
FROM sightings_log s
JOIN tigers t ON s.tiger_id = t.tiger_id
JOIN camera_stations c ON s.station_id = c.station_id
WHERE s.verified_flag = 1
ORDER BY s.sighting_timestamp ASC;