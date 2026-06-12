import requests
import time
import math

API_URL = "http://127.0.0.1:8000/ingest/spatial"

def generate_and_push_simulation():
    print("[*] Launching simulation data stream sequence loop...")
    
    # We will generate 10 frames tracking a dynamic traffic environment
    for frame_idx in range(10):
        timestamp = 20.0 + (frame_idx * 0.1) # 100ms time step offsets
        
        # Simulate an oncoming car slowly getting closer along the Y axis
        car_y_position = 25.0 - (frame_idx * 2.1) 
        
        # Adjust occlusion flags as the car slips behind simulated static blocks
        if car_y_position < 12.0:
            occlusion = "total"
        elif car_y_position < 18.0:
            occlusion = "partial_500ms"
        else:
            occlusion = "none"
            
        payload = {
            "scene_id": f"miata-run-highway-{100 + frame_idx}",
            "frame_timestamp": timestamp,
            "ego_velocity_vector": [15.5, 0.0, 0.0], # Car moving forward at 15.5 m/s
            "camera_extrinsics_rt": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.2],
                [0.0, 0.0, 1.0, -0.4]
            ],
            "detected_objects": [
                {
                    "target_id": 501,
                    "classification": "car",
                    "center_xyz": [1.8, car_y_position, 0.0],
                    "extent_lwh": [4.2, 1.7, 1.5],
                    "velocity_vector": [-20.0, -1.5, 0.0], # Target oncoming traffic
                    "occlusion_state": occlusion
                },
                {
                    "target_id": 502,
                    "classification": "motorcycle",
                    "center_xyz": [-2.5, 30.0 - (frame_idx * 0.5), 0.0],
                    "extent_lwh": [2.0, 0.8, 1.2],
                    "velocity_vector": [0.0, -5.0, 0.0],
                    "occlusion_state": "none"
                }
            ]
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                print(f"[+] Synced Frame {frame_idx+1}/10: Scene {payload['scene_id']} indexed successfully.")
            else:
                print(f"[-] Frame {frame_idx+1} Rejected by API: {response.text}")
        except Exception as e:
            print(f"[-] Transport layer failure at frame index {frame_idx}: {e}")
            
        time.sleep(0.2) # Prevent thread saturation, allow database writes to settle

    print("[+] Stream sequence run finalized. Database populated with 3D scenario records.")

if __name__ == "__main__":
    generate_and_push_simulation()
