

# About this Project 
This project is for detect and recognize emergency case of traffic. It detects traffic police and emergency cars, then it recognize the traffic police action and classify the emergency car types. It include 14 differenty type of traffic police action and 4 types of emergency cars. Architecture of emergency case handle project.
![screenshot](images/architecture_tp_ec_rec.png)

## Dependencies & Installation

Please refer to the following simple steps for installation.

```
git clone https://github.com/Ganzooo/yolov10_track_ros_gesture
cd yolov10_track_ros_gesture
pip install -r requirements.txt
```

Also our project requires ros core packadges and rospy for python3. In this project we use ros noetic packadge at Ubuntu 20.04. You can install ros noetic from official web [ROS official web](https://wiki.ros.org/ROS/Installation). There have a two line installation guide, I recommend use this guide to install [ROS two line install](https://wiki.ros.org/ROS/Installation/TwoLineInstall/).

## Prepare Test data & Weights

### Download Test Dataset
Traffic Police sample test rosbag from Drive [TP sample](https://drive.google.com/drive/folders/13V_duk5NtFBkXatJbkML-rNMW4fhzFcK?usp=sharing)

Emergency car sample test rosbag from Drive [EC sample](https://drive.google.com/drive/folders/1GKGXR9vwLHc8Lbuaw9SRQOyYqpM578df?usp=drive_link)

Model weights can be downloaded from Drive [Weights](https://drive.google.com/drive/folders/1GKGXR9vwLHc8Lbuaw9SRQOyYqpM578df?usp=drive_link). 
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

