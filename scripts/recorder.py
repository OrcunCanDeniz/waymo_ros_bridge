#!/usr/bin/env python

import rospy
import rosbag
from std_msgs.msg import String
from std_msgs.msg import Header
from geometry_msgs.msg import Point32
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from tf2_msgs.msg import TFMessage
import math
from cv_bridge import CvBridge, CvBridgeError
import os
import imp
import tensorflow as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pptk as pp

import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
from transformations import quaternion_from_euler, quaternion_from_matrix
import cv2
import pdb
lib_path = rospy.get_param("/waymo_viewer/lib_path", '')

# TODO: Change this to your own setting
#os.environ['PYTHONPATH']='/env/python:/home/user/Dataset/waymo-od'
#m = imp.find_module('waymo_open_dataset', ['/home/user/Dataset/waymo-od'])
os.environ['PYTHONPATH']='/env/python:'+lib_path
m = imp.find_module('waymo_open_dataset', [lib_path])
imp.load_module('waymo_open_dataset', m[0], m[1], m[2])

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

fields = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
    PointField('intensity', 12, PointField.FLOAT32, 1)
]


def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6
def rotationMatrixToEulerAngles(R) :

	assert(isRotationMatrix(R))
	
	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
	
	singular = sy < 1e-6
	
	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0
	
	return np.array([x, y, z])

def parse_range_image_and_camera_projection(frame):
    range_images = {}
    camera_projections = {}
    range_image_top_pose = None

    for laser in frame.lasers:

        if len(laser.ri_return1.range_image_compressed) > 0:
            range_image_str_tensor = tf.io.decode_compressed(laser.ri_return1.range_image_compressed, 'ZLIB')
            ri = open_dataset.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if laser.name == open_dataset.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                range_image_top_pose = open_dataset.MatrixFloat()
                range_image_top_pose.ParseFromString(bytearray(range_image_top_pose_str_tensor.numpy()))

            camera_projection_str_tensor = tf.io.decode_compressed(laser.ri_return1.camera_projection_compressed, 'ZLIB')
            cp = open_dataset.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name] = [cp]

        if len(laser.ri_return2.range_image_compressed) > 0:
            range_image_str_tensor = tf.io.decode_compressed(laser.ri_return2.range_image_compressed, 'ZLIB')
            ri = open_dataset.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(laser.ri_return2.camera_projection_compressed, 'ZLIB')
            cp = open_dataset.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)


    return range_images, camera_projections, range_image_top_pose

# def convert_range_image_to_point_cloud_native(frame, range_images, camera_projections, range_image_top_pose, ri_index = 0):
    # calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    # lasers = sorted(frame.lasers, key=lambda laser: laser.name)
    # points = [] 
    # cp_points = []

    # frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # # [H, W, 6]
    # range_image_top_pose_tensor = tf.reshape(tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims)
    # # [H, W, 3, 3]
    # range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],range_image_top_pose_tensor[..., 2])
    # range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    # range_image_top_pose_tensor = transform_utils.get_transform(range_image_top_pose_tensor_rotation,range_image_top_pose_tensor_translation)
    # for c in calibrations:
    #     range_image = range_images[c.name][ri_index]

    #     if len(c.beam_inclinations) == 0:
    #         beam_inclinations = range_image_utils.compute_inclination(tf.constant([c.beam_inclination_min, c.beam_inclination_max]),height=range_image.shape.dims[0])
    #     else:
    #         beam_inclinations = tf.constant(c.beam_inclinations)

    #     beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    #     extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

    #     range_image_tensor = tf.reshape(tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    #     pixel_pose_local = None
    #     frame_pose_local = None

    #     if c.name == open_dataset.LaserName.TOP:
    #         pixel_pose_local = range_image_top_pose_tensor
    #         pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
    #         frame_pose_local = tf.expand_dims(frame_pose, axis=0)
    #     range_image_mask = range_image_tensor > 0
    #     range_image_tensor = tf.where(range_image_mask, range_image_tensor,
    #                             tf.ones_like(range_image_tensor) * 1e10)
    #     range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(tf.expand_dims(range_image_tensor[..., 0], axis=0),tf.expand_dims(extrinsic, axis=0),tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),pixel_pose=pixel_pose_local,frame_pose=frame_pose_local)

    #     range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    #     points_tensor = tf.gather_nd(range_image_cartesian,tf.where(range_image_mask))

    #     cp = camera_projections[c.name][0]
    #     cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
    #     cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
    #     points.append(points_tensor.numpy())
    #     cp_points.append(cp_points_tensor.numpy())

    # return points, cp_points

def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == open_dataset.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        points.append(np.concatenate(points_single, axis=0))
        cp_points.append(np.concatenate(cp_points_single, axis=0))
        points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        points_elongation.append(np.concatenate(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation

def find_camera_label(name, camera_labels):
    for camera_label in camera_labels:
        if camera_label.name == name:
            labels = camera_label.labels
            break
    return labels

def find_camera_calibration(name, camera_calibrations):
    for calibrations in camera_calibrations:
    	if calibrations.name == name:
    		calibration = calibrations
    		break
    return calibration

def find_lidar_calibration(name, laser_calibrations):
    for calibrations in laser_calibrations:
        if calibrations.name == name:
            calibration = calibrations
            break
    return calibration

# Parameters
br = CvBridge()

vehicle = (0, 0, 255)
pedestrian = (0, 255, 0)
cyclist = (255, 0, 0)
sign = (0, 255, 255)

def talker():

    rospy.init_node("waymo_viewer", anonymous = True)
    folderpath = rospy.get_param("/waymo_viewer/folderpath", '')
    bag_path = rospy.get_param("/waymo_viewer/bag_path", '')
    files = os.listdir(folderpath)

    bLidar = True
    bLidarAnn = True
    bCamera = True
    bCameraAnn = True

    rate = rospy.Rate(1) # 10hz

    for f in files:
        fullpath = os.path.join(folderpath, f)
        BAGNAME = os.path.join(bag_path, f)
        if os.path.isfile(fullpath):
            FILENAME = fullpath
            dataset = tf.data.TFRecordDataset(FILENAME, compression_type = '')
            BAGNAME = BAGNAME.replace(".tfrecord", ".bag")
            bag = rosbag.Bag(BAGNAME, 'w')
            print(FILENAME)
            print(BAGNAME)
            print("=== start ===")

            header = Header()
            header.frame_id = '/vehicle'

            transforms = TFMessage()

            # Read frame
            for data in dataset:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))

                header.stamp.secs = int(frame.timestamp_micros / 1000000)
                header.stamp.nsecs = int((frame.timestamp_micros / 1000000.0 - header.stamp.secs)*1000000000)
                print("time stamp : ", frame.timestamp_micros)

                ### publish camera image and label
                for image in frame.images:
                    camera_image = Image()
                    camera_image.header = header

                    if frame.camera_labels:
                        labels = find_camera_label(image.name, frame.camera_labels)
                        camera_label = MarkerArray()
                        for original_label in labels:
                            label_marker = Marker()
                            label_marker.header = header
                            label_marker.type = Marker.CUBE
                            label_marker.action = Marker.ADD
                            ID = int(original_label.id[len(original_label.id)-6:len(original_label.id)].encode("hex"),32)
                            label_marker.id = np.int32(ID)
                            label_marker.pose.position.x = original_label.box.center_x
                            label_marker.pose.position.y = original_label.box.center_y
                            label_marker.scale.x = original_label.box.length
                            label_marker.scale.y = original_label.box.width
                            if original_label.type == 1:
                                label_marker.ns = "Vehicle"
                            elif original_label.type == 2:
                                label_marker.ns = "Pedestrian"
                            elif original_label.type == 3:
                                label_marker.ns = "Sign"
                            elif original_label.type == 4:
                                label_marker.ns = "Cyclist"
                            camera_label.markers.append(label_marker)
                        
                        if image.name == 1:
                            bag.write("/camera_label/front", camera_label, header.stamp)
                        elif image.name == 2:
                            bag.write("/camera_label/frontleft", camera_label, header.stamp)
                        elif image.name == 3:
                            bag.write("/camera_label/frontright", camera_label, header.stamp)
                        elif image.name == 4:
                            bag.write("/camera_label/sideleft", camera_label, header.stamp)
                        elif image.name == 5:
                            bag.write("/camera_label/sideright", camera_label, header.stamp)
                    
                    
                    calibration = find_camera_calibration(image.name, frame.context.camera_calibrations)
                    camera_image.height = calibration.height
                    camera_image.width = calibration.width
                    camera_image.encoding = "rgb8"
                    camera_image.data = np.array(tf.image.decode_jpeg(image.image)).tostring()
                    

                    # camera calibration
                    calibration_array = np.array([(calibration.extrinsic.transform[0], calibration.extrinsic.transform[1], calibration.extrinsic.transform[2], calibration.extrinsic.transform[3]),
                                                    (calibration.extrinsic.transform[4], calibration.extrinsic.transform[5], calibration.extrinsic.transform[6], calibration.extrinsic.transform[7]),
                                                    (calibration.extrinsic.transform[8], calibration.extrinsic.transform[9], calibration.extrinsic.transform[10], calibration.extrinsic.transform[11]),
                                                    (calibration.extrinsic.transform[12], calibration.extrinsic.transform[13], calibration.extrinsic.transform[14], calibration.extrinsic.transform[15])])
                    
                    camera_transform = TransformStamped()
                    camera_transform.header = Header(header.seq, header.stamp, header.frame_id)
                    rotation_matrix = np.array([(calibration_array[0][0], calibration_array[0][1], calibration_array[0][2]),
                                                (calibration_array[1][0], calibration_array[1][1], calibration_array[1][2]),
                                                (calibration_array[2][0], calibration_array[2][1], calibration_array[2][2])])
                    r,p,y = rotationMatrixToEulerAngles(rotation_matrix)
                    quaternion = quaternion_from_euler(r, p, y)
                    camera_transform.transform.rotation.x = quaternion[1]
                    camera_transform.transform.rotation.y = quaternion[2]
                    camera_transform.transform.rotation.z = quaternion[3]
                    camera_transform.transform.rotation.w = quaternion[0]
                    camera_transform.transform.translation.x = calibration_array[0][3]
                    camera_transform.transform.translation.y = calibration_array[1][3]
                    camera_transform.transform.translation.z = calibration_array[2][3]
                    

                    if image.name == 1:
                        camera_transform.child_frame_id = "/camera_front"
                        bag.write("/camera_image/front", camera_image, header.stamp)
                    elif image.name == 2:
                        camera_transform.child_frame_id = "/camera_frontleft"
                        bag.write("/camera_image/frontleft", camera_image, header.stamp)
                    elif image.name == 3:
                        camera_transform.child_frame_id = "/camera_frontright"
                        bag.write("/camera_image/frontright", camera_image, header.stamp)
                    elif image.name == 4:
                        camera_transform.child_frame_id = "/camera_sideleft"
                        bag.write("/camera_image/sideleft", camera_image, header.stamp)
                    elif image.name == 5:
                        camera_transform.child_frame_id = "/camera_sideright"
                        bag.write("/camera_image/sideright", camera_image, header.stamp)
                    
                    transforms.transforms.append(camera_transform)


                ### lidar visualization
                (range_images, camera_projections, range_image_top_pose) = parse_range_image_and_camera_projection(frame)
                # Calculate the point cloud
                points, cp_points, points_NLZ, points_intensity, points_elongation = convert_range_image_to_point_cloud(frame, range_images, camera_projections,range_image_top_pose)
                # pdb.set_trace()
                # publish lidar pointcloud and label
                for index, lidar_points in enumerate(points):
                    lidar_msg = PointCloud2()
                    lidar_msg.header = header
                    lidar_name = index+1
                    lidar_data = []
                    
                    intensities = points_intensity[index]
                    
                    for intensity_idx, point in enumerate(points[index]):
                        lidar_data.append([point[0], point[1], point[2], intensities[intensity_idx]])
                    cloud_arr = np.array(lidar_data)
                    
                    lidar_msg.height = cloud_arr.shape[0]
                    lidar_msg.width = 1
                    lidar_msg.fields = fields
                    lidar_msg.is_bigendian = False # assumption
                    lidar_msg.point_step = 16
                    # pdb.set_trace()
                    # lidar_msg.point_step = cloud_arr.dtype.itemsize
                    lidar_msg.row_step = lidar_msg.point_step*cloud_arr.shape[1]
                    lidar_msg.is_dense = False
                    lidar_msg.data = cloud_arr.tostring()
                    
                    calibration = find_lidar_calibration(lidar_name, frame.context.laser_calibrations)

                    # lidar calibration
                    calibration_array = np.array([(calibration.extrinsic.transform[0], calibration.extrinsic.transform[1], calibration.extrinsic.transform[2], calibration.extrinsic.transform[3]),
                                                    (calibration.extrinsic.transform[4], calibration.extrinsic.transform[5], calibration.extrinsic.transform[6], calibration.extrinsic.transform[7]),
                                                    (calibration.extrinsic.transform[8], calibration.extrinsic.transform[9], calibration.extrinsic.transform[10], calibration.extrinsic.transform[11]),
                                                    (calibration.extrinsic.transform[12], calibration.extrinsic.transform[13], calibration.extrinsic.transform[14], calibration.extrinsic.transform[15])])                      

                    lidar_transform = TransformStamped()
                    lidar_transform.header = Header(header.seq, header.stamp, header.frame_id)
                    rotation_matrix = np.array([(calibration_array[0][0], calibration_array[0][1], calibration_array[0][2]),
                                                (calibration_array[1][0], calibration_array[1][1], calibration_array[1][2]),
                                                (calibration_array[2][0], calibration_array[2][1], calibration_array[2][2])])
                    r,p,y = rotationMatrixToEulerAngles(rotation_matrix)
                    quaternion = quaternion_from_euler(r, p, y)
                    lidar_transform.transform.rotation.x = quaternion[1]
                    lidar_transform.transform.rotation.y = quaternion[2]
                    lidar_transform.transform.rotation.z = quaternion[3]
                    lidar_transform.transform.rotation.w = quaternion[0]
                    lidar_transform.transform.translation.x = calibration_array[0][3]
                    lidar_transform.transform.translation.y = calibration_array[1][3]
                    lidar_transform.transform.translation.z = calibration_array[2][3]
                    
                    
                    if lidar_name == 1:
                        bag.write("/lidar_pointcloud/top", lidar_msg, header.stamp)
                        lidar_transform.child_frame_id = "/lidar_top"
                    elif lidar_name == 2:
                        bag.write("/lidar_pointcloud/front", lidar_msg, header.stamp)
                        lidar_transform.child_frame_id = "/lidar_front"
                    elif lidar_name == 3:
                        bag.write("/lidar_pointcloud/left", lidar_msg, header.stamp)
                        lidar_transform.child_frame_id = "/lidar_left"
                    elif lidar_name == 4:
                        bag.write("/lidar_pointcloud/right", lidar_msg, header.stamp)
                        lidar_transform.child_frame_id = "/lidar_right"
                    elif lidar_name == 5:
                        bag.write("/lidar_pointcloud/rear", lidar_msg, header.stamp)
                        lidar_transform.child_frame_id = "/lidar_rear"
                    transforms.transforms.append(lidar_transform)
                    
                    if lidar_name == 1:
                        lidar_label = MarkerArray()
                        for index, laser_label in enumerate(frame.laser_labels):
                            label_marker = Marker()
                            label_marker.header = header
                            label_marker.type = Marker.CUBE
                            label_marker.action = Marker.ADD
                            ID = int(laser_label.id[len(laser_label.id)-8:len(laser_label.id)].encode("hex"),16)
                            label_marker.id = np.int32(ID)
                            label_marker.lifetime = rospy.Duration(0.1)
                            label_marker.pose.position.x = laser_label.box.center_x
                            label_marker.pose.position.y = laser_label.box.center_y
                            label_marker.pose.position.z = laser_label.box.center_z
                            quaternion = quaternion_from_euler(0, 0, laser_label.box.heading)
                            label_marker.pose.orientation.x = quaternion[1]
                            label_marker.pose.orientation.y = quaternion[2]
                            label_marker.pose.orientation.z = quaternion[3]
                            label_marker.pose.orientation.w = quaternion[0]
                            label_marker.scale.x = laser_label.box.length
                            label_marker.scale.y = laser_label.box.width
                            label_marker.scale.z = laser_label.box.height
                            label_marker.color.a = 0.5
                            if laser_label.type == 1 :
                                label_marker.color.b = vehicle[0]
                                label_marker.color.g = vehicle[1]
                                label_marker.color.r = vehicle[2]
                                label_marker.ns = "Vehicle"

                            elif laser_label.type == 2 :
                                label_marker.color.b = pedestrian[0]
                                label_marker.color.g = pedestrian[1]
                                label_marker.color.r = pedestrian[2]
                                label_marker.ns = "Pedestrian"

                            elif laser_label.type == 3 :
                                label_marker.color.b = sign[0]
                                label_marker.color.g = sign[1]
                                label_marker.color.r = sign[2]
                                label_marker.ns = "Sign"

                            elif laser_label.type == 4 :
                                label_marker.color.b = cyclist[0]
                                label_marker.color.g = cyclist[1]
                                label_marker.color.r = cyclist[2]
                                label_marker.ns = "Cyclist"
                            lidar_label.markers.append(label_marker)
                        bag.write("/lidar_label", lidar_label, header.stamp)

                # Publish transform
                
                transform = TransformStamped()
                transform.header = Header(header.seq, header.stamp, "/global")
                rotation_matrix = np.array([(frame.pose.transform[0], frame.pose.transform[1], frame.pose.transform[2]),
                                            (frame.pose.transform[4], frame.pose.transform[5], frame.pose.transform[6]),
                                            (frame.pose.transform[8], frame.pose.transform[9], frame.pose.transform[10])])
                r,p,y = rotationMatrixToEulerAngles(rotation_matrix)
                quaternion = quaternion_from_euler(r, p, y)
                transform.transform.rotation.x = quaternion[1]
                transform.transform.rotation.y = quaternion[2]
                transform.transform.rotation.z = quaternion[3]
                transform.transform.rotation.w = quaternion[0]
                transform.transform.translation.x = frame.pose.transform[3]
                transform.transform.translation.y = frame.pose.transform[7]
                transform.transform.translation.z = frame.pose.transform[11]
                transform.child_frame_id = "/vehicle"
                transforms.transforms.append(transform)

                bag.write("/tf", transforms, header.stamp) 
                
            bag.close()
            print(FILENAME)
            print("=== end ===")                   

    rate.sleep()

if __name__ == '__main__': 
    try:
	    talker()
    except  rospy.ROSInterruptException:
	    pass
