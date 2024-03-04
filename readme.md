* Download the model and put it in the "models" folder. 

  ```bash
  cd models
  wget https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip
  unzip antelopev2.zip
  ```

  At this point, the model path is "models/antelopev2/*.onnx"

* Set up a conda environment:

  ```bash
  conda create --name mad_face python=3.10
  conda activate mad_face
  pip install insightface onnxruntime-gpu decord
  conda install -c anaconda cudnn cudatoolkit
  ```

* Test the environment:

  ```bash
  python test.py
  ```

  You will not see any warnings, and the output should be `4`.

* Modify lines 54 to 56 in the `main.py` file

  * Each item in `video_list` is a tuple containing two elements. The first element is the video's ID, and the second element is the video's path. e.g. 

    ```python
    video_list = [("9952", '/path/to/video_9952'), ("9846", '/path/to/video_9846'), ("...", '...')]
    ```

  * `worker_count` represents the number of threads. The videos in the `video_list` will be evenly distributed among each thread for feature extraction.

  * `n_gpus` represents the number of GPUs in use. This implies that there  will be `worker_count / n_gpus` models on each GPU to extract features.

* Run 

  ```bash
  python main.py
  ```

  

The features extracted from each video will be written to a JSON file in the "result" folder. On our machine (`worker_count=8, 2 A100 GPUs`), each worker can compute  approximately 10 frames per second. This implies that a single worker  might take over a dozen minutes to complete the computation for a 2-hour video.

If the feature extraction is completed, please provide me with the download instructions. Thank you very much for your efforts.



Bin Huang

huangb23@mails.tsinghua.edu.cn