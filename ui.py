import streamlit as st
import requests
import json

# Force layout alignment to maximize terminal widescreen layouts
st.set_page_config(layout="wide", page_title="Spatial Telemetry Dashboard")

API_BASE = "http://127.0.0.1:8000"

st.title("🛰️ Spatial Telemetry & Scenario Retrieval Engine")
st.caption("Local 3D Scene Discovery, Volumetric Processing & State Seeding Dashboard")

# Split our interface workspace down the center into 2 distinct columns
col1, col2 = st.columns(2)

with col1:
    st.header("📥 Ingestion Workspace")
    st.subheader("Stream Telemetry Configuration Package")
    
    # Pre-populate a standard 3D sample frame layout for easy workspace testing
    sample_payload = {
        "scene_id": "miata-sequence-042",
        "frame_timestamp": 12.450,
        "ego_velocity_vector": [11.2, 0.0, -0.05],
        "camera_extrinsics_rt": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.2],
            [0.0, 0.0, 1.0, -0.4]
        ],
        "detected_objects": [
            {
                "target_id": 105,
                "classification": "car",
                "center_xyz": [2.5, 14.2, -0.1],
                "extent_lwh": [4.5, 1.8, 1.4],
                "velocity_vector": [-8.5, 0.2, 0.0],
                "occlusion_state": "partial_500ms"
            },
            {
                "target_id": 106,
                "classification": "pedestrian",
                "center_xyz": [-1.8, 8.5, -0.2],
                "extent_lwh": [0.6, 0.6, 1.7],
                "velocity_vector": [0.0, 1.1, 0.0],
                "occlusion_state": "none"
            }
        ]
    }
    
    json_input = st.text_area(
        "Structured Spatial JSON Input Contract:",
        value=json.dumps(sample_payload, indent=2),
        height=400
    )
    
    if st.button("🚀 Push Telemetry to Database Cluster"):
        try:
            parsed_payload = json.loads(json_input)
            with st.spinner("Executing matrix serialization and pgvector indexing..."):
                response = requests.post(f"{API_BASE}/ingest/spatial", json=parsed_payload)
                
            if response.status_code == 200:
                res_data = response.json()
                st.success(f"Successfully indexed sequence: {res_data['scene_indexed']} at timestamp {res_data['frame_timestamp']}s")
            else:
                st.error(f"Backend Server Error Flag: {response.text}")
        except json.JSONDecodeError:
            st.error("Data Verification Failure: Input block is not a valid JSON string structure.")
        except Exception as e:
            st.error(f"Transport Connection Failed: {e}")

with col2:
    st.header("🔍 Conversational Query Workspace")
    st.subheader("Scenario Matrix Constraint Extraction")
    
    query_prompt = st.text_input(
        "Search Query:",
        placeholder="e.g., Find oncoming vehicles under occlusion closer than 15 meters"
    )
    
    if st.button("⚡ Run Spatial Search Strategy"):
        if query_prompt:
            with st.spinner("Computing query embeddings and scanning cosine distance matrices..."):
                try:
                    response = requests.post(f"{API_BASE}/query/scenario", json={"prompt": query_prompt})
                    
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No response content returned.")
                        st.subheader("🤖 Synthesized Engineering Output:")
                        st.markdown(answer)
                    else:
                        st.error(f"Backend Query Processing Exception: {response.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with API server: {e}")
        else:
            st.warning("Please specify a constraint scenario query sequence first.")
