import time

import rospy

from ultralytics import YOLOv10

print('Yolov10')
detection_model = YOLOv10("/mnt/data_main/NETWORK/18_TrafficPolice_Inference/YOLOv10-Track-Detect-Classify-Rec/data/weight/detection/yolov10b_keti_tp_0_889.pt")

print(detection_model)

rospy.init_node("ultralytics", anonymous=True)
#time.sleep(1)

from sensor_msgs.msg import Image

print('Rosb Pub')
det_image_pub = rospy.Publisher("/camera/image_raw/compressed", Image, queue_size=5)

import ros_numpy

def callback(data):
    """Callback function to process image and publish annotated images."""
    array = ros_numpy.numpify(data)
    if det_image_pub.get_num_connections():
        det_result = detection_model(array)
        det_annotated = det_result[0].plot(show=False)
        print(det_result)
        #det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

print('Ros Sub')
rospy.Subscriber("/camera/image_raw/compressed_det", Image, callback)

while True:
    rospy.spin()