import asyncio
import websockets
import json
import numpy as np
import base64
import io
import os
import keyboard
import nest_asyncio
import re
import cv2

nest_asyncio.apply()

MESSAGE_FILE = "messages.txt"

def get_next_index():
    if not os.path.exists(MESSAGE_FILE):
        return 1
    with open(MESSAGE_FILE, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        indices = [int(line.split('.')[0]) for line in lines if line.strip() and '.' in line]
        return max(indices) + 1 if indices else 1

def get_last_sign_user_message():
    try:
        with open(MESSAGE_FILE, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "sign_user:" in line:
                    clean_message = line.replace("sign_user:", "").strip()
                    clean_message = re.sub(r'\d+', '', clean_message)
                    return clean_message.strip()
    except FileNotFoundError:
        print("messages.txt not found.")
    return None

def visualize_skeleton(keypoints_sequence, output_path):
    frame_h, frame_w, fps = 720, 1280, 40
    video_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    POSE_CONNECTIONS = [(11,12),(12,14),(14,16),(11,13),(13,15),(11,23),(12,24),
                        (23,24),(23,25),(24,26),(25,27),(27,29),(26,28),(28,30),
                        (23,11),(24,12),(11,12),(23,24)]
    HAND_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                        (0,17),(17,18),(18,19),(19,20)]
    FACE_CONNECTIONS = [(i, i+1) for i in range(467)]

    def draw_connections(frame, kpts, connections, color, thickness=3):
        for p1, p2 in connections:
            if all(0 <= idx < len(kpts) for idx in [p1, p2]):
                pt1 = (int(kpts[p1][0] * frame_w), int(kpts[p1][1] * frame_h))
                pt2 = (int(kpts[p2][0] * frame_w), int(kpts[p2][1] * frame_h))
                cv2.line(frame, pt1, pt2, color, thickness)

    for keypoints in keypoints_sequence:
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        keypoints = keypoints.reshape(-1, 3)

        draw_connections(frame, keypoints[0:468], FACE_CONNECTIONS, (200,200,200), 1) 
        draw_connections(frame, keypoints[468:501], POSE_CONNECTIONS, (0, 165, 255), 4) 
        draw_connections(frame, keypoints[501:522], HAND_CONNECTIONS, (255, 215, 0), 3) 
        draw_connections(frame, keypoints[522:543], HAND_CONNECTIONS, (30, 144, 255), 3) 

        for x, y, _ in keypoints:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x * frame_w), int(y * frame_h)), 4, (50,205,50), -1)

        video_out.write(frame)
    
    video_out.release()
    

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Skeleton Animation', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

async def check_c_key(websocket):
    while True:
        await asyncio.sleep(0.1)
        if keyboard.is_pressed("c"):
            sign_user_message = get_last_sign_user_message()
            if sign_user_message:
                response_to_production = {"sign_user_message": sign_user_message}
                await websocket.send(json.dumps(response_to_production))
                print(f"✅ Sent sign_user message: {sign_user_message}")
                await asyncio.sleep(1)

async def translation_server(websocket):
    print("✅ Production connected.")
    asyncio.create_task(check_c_key(websocket))

    try:
        async for received_data in websocket:
            print(f"🛠 DEBUG: Received Raw Data: {received_data}")
            data = json.loads(received_data)
            question = data.get("question", "")
            npy_base64 = data.get("npy", None)

            print(f"🟢 Received question from normal_user: {question}")
            index = get_next_index()

            if npy_base64:
                npy_bytes = base64.b64decode(npy_base64)
                npy_array = np.load(io.BytesIO(npy_bytes))
                np.save("received.npy", npy_array)
                print(f"✅ Received NPY file (Shape: {npy_array.shape})")
                output_video = "final_skeleton_output.mp4"
                visualize_skeleton(npy_array, output_video)
                play_video(output_video)

            with open(MESSAGE_FILE, "a", encoding="utf-8") as f:
                f.write(f"{index}. normal_user: {question}\n")

            answer = f"Processed message #{index}"
            response = {"answer": answer}
            await websocket.send(json.dumps(response))
            print(f"✅ Sent response to Production: {answer}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

async def main():
    async with websockets.serve(translation_server, "192.168.84.139", 8765, max_size=50 * 1024 * 1024, ping_interval=None):
        print("✅ Translation Server running at ws://192.168.84.139:8765 (Max size: 50MB)")
        await asyncio.Future()

asyncio.run(main())
