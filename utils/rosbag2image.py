import rosbag
import cv2
from cv_bridge import CvBridge
import os

# Initialize CvBridge
bridge = CvBridge()

# Path to your ROS bag file and output directory
bag_file = '/media/tt/data/dataset/ros_data/certificate/tp/tp_police/k5_50m_2023-11-13-16-56-12.bag'
output_dir = '/media/tt/data/dataset/ros_data/certificate/tp/emergency_car/k5_back_emergency_car'
file_name_type = 'k5_back_emergency_car'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

     
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

# Initialize CvBridge
bridge = CvBridge()

# Sampling interval in seconds (e.g., 0.5 seconds)
sampling_interval = 0.5  # Adjust this value as needed

# Variable to keep track of the last saved timestamp
frame_count = 0
last_saved_time = 0

# Callback function to process the compressed image
def compressed_image_callback(msg):
    try:
        
        current_time = msg.header.stamp.to_sec()
        #if (current_time - last_saved_time) >= sampling_interval and frame_count > 0:
            
            # Update the last saved time
        #    last_saved_time = current_time
            
        # Convert the compressed image data to a numpy array
        np_arr = np.frombuffer(msg.data, np.uint8)

        # Decode the compressed image (JPEG/PNG) into an OpenCV image (BGR format)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Display the image (optional)
        #cv2.imshow("Decompressed Image", cv_image)
        #cv2.waitKey(1)

        filename = f"image_{file_name_type}_{current_time:.9f}.png"
        cv2.imwrite(os.path.join(output_dir,filename), cv_image)
        print(f"Saved {filename}")
        
        #frame_count = frame_count + 1
    except Exception as e:
        print(f"Error processing image: {e}")

def main():
    # Initialize ROS node
    rospy.init_node('compressed_image_subscriber', anonymous=True)

    # Subscribe to the compressed image topic
    rospy.Subscriber('/camera1/camera/image_raw/compressed', CompressedImage, queue_size=100, callback=compressed_image_callback)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    main()