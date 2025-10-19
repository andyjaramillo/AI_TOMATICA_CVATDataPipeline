
# IFSA AI_Tomatica People Detector
 
People Detector is a Python script that processes videos as input and performs individual people detection, tracking, and counting. It uses [YOLOv5](https://github.com/ultralytics/yolov5), a state-of-the-art deep learning model for object detection, and [motpy](https://github.com/wmuron/motpy), a multi-object tracking library. It then displays bounding boxes around each person, assigns unique IDs, and shows the count of people in the video frame.

## Run People Detector

1. Clone the repository
   
   `git clone https://github.com/scimone/peopledetector.git`
2. Install required packages with a conda environment

   ```
   conda create -n person-detection
   conda activate person-detection
   conda install --file requirements.txt
   ```
3. Place your video file in the `data` folder.
4. Change into the project directory.
5. Run the main script with `python3 main.py`.
6. Output will be saved as `image_{count}.png` in `output` folder.
