import os
import subprocess
import time
import socket
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg2
from pgvector.psycopg2 import register_vector
import ollama

app = FastAPI(title="Spatial Telemetry Engine", description="Local 3D Scene Discovery & Volumetric Retrieval Platform")

# Configured to target your newly initialized PostgreSQL layer
DB_PARAMS = {
    "dbname": "spatial_vector_db",
    "user": "postgres",
    "password": "",
    "host": "localhost",
    "port": "5432"
}

LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

def bootstrap_ollama():
    """Checks if the local Ollama port is open. If closed, automates engine boot."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_running = s.connect_ex(('127.0.0.1', 11434)) == 0

    if not is_running:
        print("[*] Ollama daemon is offline. Initializing automatic background boot thread...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)  # Allow time slice for socket allocation
        except Exception as e:
            print(f"[-] Auto-boot failed. Attempting alternative systemd invoke: {e}")
            subprocess.Popen(["sudo", "systemctl", "start", "ollama"])
            time.sleep(3)

    try:
        local_models = [m['model'] for m in ollama.list().get('models', [])]
        if f"{EMBED_MODEL}:latest" not in local_models and EMBED_MODEL not in local_models:
            print(f"[*] Fetching required embedding matrix: {EMBED_MODEL}")
            ollama.pull(EMBED_MODEL)
        if f"{LLM_MODEL}:latest" not in local_models and LLM_MODEL not in local_models:
            print(f"[*] Fetching required synthesis matrix: {LLM_MODEL}")
            ollama.pull(LLM_MODEL)
        print("[+] Ollama execution layer online and verified.")
    except Exception as e:
        print(f"[-] Target verification error: {e}. Confirm manually via 'ollama list'")

# Fire daemon automation layer
bootstrap_ollama()

def init_db():
    """Initializes extension registers and spatial telemetry store tables."""
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    # Mutate table schema to explicitly hold structural physical states
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spatial_scene_store (
            id serial PRIMARY KEY,
            scene_id text NOT NULL,
            frame_timestamp real NOT NULL,
            ego_velocity_vector real[] NOT NULL,
            raw_telemetry_json jsonb NOT NULL,
            spatial_geometry_embedding vector(768)
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

init_db()

# --- Pydantic Validation Models for Structural Telemetry Ingestion (V2 Standards Complete) ---

class BoundingBox3D(BaseModel):
    target_id: int
    classification: str
    center_xyz: List[float] = Field(..., min_length=3, max_length=3)  # [X, Y, Z] relative coordinates
    extent_lwh: List[float] = Field(..., min_length=3, max_length=3)  # [Length, Width, Height]
    velocity_vector: List[float] = Field(..., min_length=3, max_length=3) # [Vx, Vy, Vz]
    occlusion_state: str # e.g., "none", "partial_500ms", "total"

class SpatialFramePayload(BaseModel):
    scene_id: str
    frame_timestamp: float
    ego_velocity_vector: List[float] = Field(..., min_length=3, max_length=3) # [Vx, Vy, Vz]
    camera_extrinsics_rt: List[List[float]] # 3x4 or 4x4 matrix transformation rows
    detected_objects: List[BoundingBox3D]

class QueryPayload(BaseModel):
    prompt: str

# --- Spatial Math Serialization Utility ---

def serialize_spatial_frame(payload: SpatialFramePayload) -> str:
    """
    Converts multi-dimensional matrix inputs and coordinate tracking telemetry
    into a dense structural string representation for the embedding layer.
    """
    ego_speed = (sum(v**2 for v in payload.ego_velocity_vector))**0.5

    # Base structural metadata anchor
    structural_string = (
        f"Sequence frame token: {payload.scene_id}. Physical Timestamp offsets: {payload.frame_timestamp:.3f}s. "
        f"Ego vehicle linear velocity state magnitude: {ego_speed:.2f} m/s. "
    )

    # Serialize target bounding box trajectories relative to ego origin
    for idx, box in enumerate(payload.detected_objects):
        distance_euclidean = (sum(c**2 for c in box.center_xyz))**0.5
        structural_string += (
            f"Object index: {idx} [ID: {box.target_id}] Type: {box.classification}. "
            f"Position vector coordinates X:{box.center_xyz[0]:.2f} Y:{box.center_xyz[1]:.2f} Z:{box.center_xyz[2]:.2f}. "
            f"Radial distance to target: {distance_euclidean:.2f} meters. "
            f"Extent boundaries length-width-height: {' '.join(map(str, box.extent_lwh))}. "
            f"Trajectory velocity component vectors Vx:{box.velocity_vector[0]:.2f} Vy:{box.velocity_vector[1]:.2f}. "
            f"Occlusion status flag: {box.occlusion_state}. "
        )

    return structural_string

# --- REST Endpoints ---

@app.post("/ingest/spatial")
async def ingest_spatial_telemetry(payload: SpatialFramePayload):
    try:
        # Convert raw numerical matrices into a dense semantic topology map
        serialized_context = serialize_spatial_frame(payload)

        # Generate spatial feature embedding coordinates
        response = ollama.embed(model=EMBED_MODEL, input=serialized_context)
        spatial_vector = response['embeddings'][0] if 'embeddings' in response else response['embedding']

        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        register_vector(conn)

        # Store explicit spatial variables alongside raw structural jsonb tracking logs
        cursor.execute(
            """
            INSERT INTO spatial_scene_store
            (scene_id, frame_timestamp, ego_velocity_vector, raw_telemetry_json, spatial_geometry_embedding)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (
                payload.scene_id,
                payload.frame_timestamp,
                payload.ego_velocity_vector,
                json.dumps(payload.dict()),
                spatial_vector
            )
        )

        conn.commit()
        cursor.close()
        conn.close()

        print(f"[+] Successfully indexed tracking metrics for sequence frame: {payload.scene_id}")
        return {"status": "success", "scene_indexed": payload.scene_id, "frame_timestamp": payload.frame_timestamp}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/scenario")
async def query_scenario_pipeline(payload: QueryPayload):
    try:
        # Embed the spatial tracking constraint request
        response = ollama.embed(model=EMBED_MODEL, input=payload.prompt)
        query_vector = response['embeddings'][0] if 'embeddings' in response else response['embedding']

        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        register_vector(conn)

        # Query the database for the 3 closest spatial layouts using cosine distance
        cursor.execute(
            """
            SELECT scene_id, frame_timestamp, raw_telemetry_json
            FROM spatial_scene_store
            ORDER BY spatial_geometry_embedding <=> %s::vector
            LIMIT 3;
            """,
            (query_vector,)
        )
        records = cursor.fetchall()
        cursor.close()
        conn.close()

        if not records:
            return {"answer": "No indexed scene matrices discovered in spatial database layers."}

        # Build context payloads by isolating individual target properties into an explicit schema map
        context_blocks = []
        for row in records:
            scene_id = row[0]
            timestamp = row[1]
            raw_json = row[2]

            objects_summary = ""
            for obj in raw_json.get("detected_objects", []):
                objects_summary += (
                    f"- TARGET ID {obj['target_id']} Class: [{obj['classification'].upper()}]\n"
                    f"  Position [X,Y,Z]: {obj['center_xyz']}\n"
                    f"  Size [L,W,H]: {obj['extent_lwh']}\n"
                    f"  Velocity [Vx,Vy,Vz]: {obj['velocity_vector']}\n"
                    f"  Visibility State: {obj['occlusion_state']}\n"
                )

            block = (
                f"DATA SOURCE NODE: {scene_id} at {timestamp:.2f}s\n"
                f"OBJECT LOGS:\n{objects_summary}"
            )
            context_blocks.append(block)

        context_payload = "\n---\n".join(context_blocks)

        # Rigidly lock down the small LLM's role to prevent semantic merging and field re-writing
        system_instructions = (
            "You are a deterministic Autonomous Vehicle Data Extraction Component. You do not reason or theorize, "
            "you do not correct source classifications, and you never guess properties.\n\n"
            "CRITICAL COMMANDS:\n"
            "1. Output ONLY exact parameters directly printed in the OBJECT LOGS context. If a target is logged as a motorcycle, "
            "write its classification explicitly as motorcycle. NEVER alter it or label it as a pedestrian.\n"
            "2. Keep numeric values completely intact. If velocities are logged as meters per second, output them in m/s, never mph.\n"
            "3. Do not cross-contaminate properties. Keep every target ID perfectly mapped to its own structural location vectors.\n"
            "4. For the single closest object matching the user query prompt constraints, output its tracking parameter block using "
            "this exact markdown notation:\n"
            "$$x = [X, Y, V_x, V_y]^T$$\n"
            "followed by the exact numbers from that specific target's log file entry lines.\n\n"
            f"=== RETRIEVED TELEMETRY CONTEXT BANDS ===\n{context_payload}\n========================================="
        )

        chat_res = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": payload.prompt}
            ]
        )

        return {"answer": chat_res['message']['content']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
