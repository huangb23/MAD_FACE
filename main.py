import base64
import json
import math
import os
import cv2
import numpy as np
import decord
from insightface.app import FaceAnalysis
from tqdm import tqdm
import concurrent.futures

def extract_frames(video_path, output_fps=1):
    video = decord.VideoReader(video_path)
    fps = video.get_avg_fps()
    total_frames = len(video)
    frame_intervals = int(fps / output_fps)
    sampled_indices = list(range(0, total_frames, frame_intervals))
    extracted_frames = video.get_batch(sampled_indices).asnumpy()
    return extracted_frames

def np_to_str(x):
    return base64.b64encode(x.tobytes()).decode('utf-8')

def str_to_np(x):
    return np.frombuffer(base64.b64decode(x), dtype=np.float32) 

def analysis(video_list, gpu_id):
    app = FaceAnalysis(name='antelopev2', root='.', allowed_modules=['detection', 'recognition'], providers=[('CUDAExecutionProvider',  {'device_id': gpu_id}), 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    for id, video_path in tqdm(video_list):
        output_path = f"result/{id}.json"
        # if os.path.isfile(output_path): continue

        frames = extract_frames(video_path, output_fps=1)
        print(id, "#frames:", len(frames))
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

        result = []
        for frame in frames:
            face_info = app.get(frame)
            face_info = [{
                'bbox': [str(i) for i in x['bbox']],
                'kps': [[str(j) for j in i] for i in x['kps']],
                'det_score': str(x['det_score']),
                'embedding': np_to_str(x['embedding'])
            } for x in face_info]
            result.append(face_info)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)

if __name__ == '__main__':
    video_list = [("9952", '/path/to/video_9952'), ("9846", '/path/to/video_9846'), ("...", '...')]
    worker_count = 8
    n_gpus = 2

    worker_per_gpu = math.ceil(worker_count / n_gpus)
    video_list_parts = [video_list[i::worker_count] for i in range(worker_count)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(analysis, video_list_parts[i], gpu_id=i//worker_per_gpu) for i in range(worker_count)]
        concurrent.futures.wait(futures)