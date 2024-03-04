import cv2
import numpy as np
import PIL
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='antelopev2', root='.', allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

image = PIL.Image.open('example_image.webp').convert("RGB")
face_info = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
print(len(face_info)) # 4