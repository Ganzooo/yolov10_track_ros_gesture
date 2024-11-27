<div align="center">

# About this Project 


## Dependencies & Installation

Please refer to the following simple steps for installation.

```
git clone https://github.com/Ganzooo/yolov10_track_ros_gesture
cd yolov10_track_ros_gesture
pip install -r requirements.txt
```

## Prepare Test data & Weights

### Download Test Dataset
Traffic Police sample test rosbag from Drive [AIS2024](https://drive.google.com/drive/folders/13V_duk5NtFBkXatJbkML-rNMW4fhzFcK?usp=sharing)

Emergency car sample test rosbag from Drive [DIV2K](https://drive.google.com/drive/folders/1GKGXR9vwLHc8Lbuaw9SRQOyYqpM578df?usp=drive_link)

Model weights can be downloaded from Drive [DIV2K](https://drive.google.com/drive/folders/1GKGXR9vwLHc8Lbuaw9SRQOyYqpM578df?usp=drive_link). 
Copy all weight folder to ./data/weight/ folder.

### Run rosbag file and visualization
You need to open several seperated terminals. 

Terminal 1 (run roscore):
```
roscore
```

Terminal 2 (play rosbag):
```
cd /rosbag_path/
rosbag play -i name_of_rosbag.bag
```

Terminal 3 (rviz for visualization):
```
rviz
```

## Testing
You can run it at command line.

```
python ./united_trafficpolice_emergency_case_datection.py
```

## Check Result
RVIZ test result.
Traffic plolice action recognition result(Top: Input rosbag, Bottom: Detected and Recognized rosbag)
![screenshot](images/Action_rec_result.png)

Emergency car detection result(Top: Input rosbag, Bottom: Detected and Classified rosbag)
![screenshot](images/EC_result.png)

