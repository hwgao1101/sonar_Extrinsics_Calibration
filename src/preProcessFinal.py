import os

import numpy as np
import rosbag
import rospy
import cv2
import pickle

'''
2024-01-16最新搜集的标定数据进行处理
'''
def save_image(bag_file_path, startTime, endTime, save_image_path):
    # bag_file_path = '/ubuntu/sonarCalibration/2023-12-19-21-58-37.bag'
    bag = rosbag.Bag(bag_file_path)
    image_topic = '/sonar_oculus_node/image'
    imageTime = []
    for i in range(startTime.shape[0]):
        image_data = bag.read_messages(topics=[image_topic], start_time=rospy.Time(startTime[i]), end_time=rospy.Time(endTime[i]))
        eachTime = []
        j = 1
        for topic, msg, t in image_data:
            eachTime.append(t.nsecs)
            img = np.frombuffer(msg.data, np.uint8)
            img = img.reshape((msg.height, msg.width, 3))
            # cv2.imshow('image', img)
            # cv2.waitKey(0)
            os.makedirs(f'{save_image_path}/{i + 1}', exist_ok=True)
            cv2.imwrite(f'{save_image_path}/{i + 1}/img{j}.jpg', img)
            j += 1

        imageTime.append(np.stack(eachTime))

    bag.close()
    return imageTime

def save_raw_image(bag_file_path, startTime, endTime, save_raw_image_path):
    # bag_file_path = '/ubuntu/sonarCalibration/2023-12-19-21-58-37.bag'
    bag = rosbag.Bag(bag_file_path)
    rawImageTime = []
    for i in range(startTime.shape[0]):
        raw_image_data = bag.read_messages(topics=['/sonar_oculus_node/raw_img'], start_time=rospy.Time(startTime[i]), end_time=rospy.Time(endTime[i]))
        eachTime = []
        j = 1
        for topic, msg, t in raw_image_data:
            eachTime.append(t.nsecs)
            img = np.frombuffer(msg.data, np.uint8)
            img = img.reshape((msg.height, msg.width))
            # img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # img_color[:, :, 1:] = 0
            # cv2.imshow('raw_image', img_color)
            # cv2.waitKey(0)
            os.makedirs(f'{save_raw_image_path}/{i + 1}', exist_ok=True)
            cv2.imwrite(f'{save_raw_image_path}/{i + 1}/img{j}.jpg', img)
            j += 1
        rawImageTime.append(eachTime)
    bag.close()
    return rawImageTime

def getSonarFixatorPose(bag_file_path, startTime, endTime):
    '''
    把时间戳插值之后的SonarFixator动捕位姿返回回去
    :param bag_file_path:
    :param startTime:
    :param endTime:
    :return:
    '''
    # 动捕数据的单位变换
    transUnit = 1e-3
    bag = rosbag.Bag(bag_file_path)
    transormMatrix = []  # 用来存放所有的SonarFixator的变换矩阵
    sonarFixatorTime = []  # 用来存放时间戳
    for i in range(startTime.shape[0]):
        sonarFixatorData = bag.read_messages(topics=['/vrpn_client_node/TrackerSonar/pose'], start_time=rospy.Time(startTime[i]), end_time=rospy.Time(endTime[i]))
        eachMatrix = []
        eachTime = []
        for topic, msg, t in sonarFixatorData:
            eachTime.append(t.nsecs)
            # 下面由四元数计算旋转矩阵
            rotation_quaternion = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            temp_Matrix = np.eye(4, 4)
            x = rotation_quaternion[0]
            y = rotation_quaternion[1]
            z = rotation_quaternion[2]
            w = rotation_quaternion[3]
            temp_Matrix[:3, :3] = np.array([
                [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
            ])

            # 计算平移矩阵
            translation_vector = np.array([msg.pose.position.x * transUnit, msg.pose.position.y * transUnit, msg.pose.position.z * transUnit])
            temp_Matrix[:3, 3] = translation_vector
            eachMatrix.append(temp_Matrix)

        transormMatrix.append(eachMatrix)
        sonarFixatorTime.append(eachTime)

    return transormMatrix, sonarFixatorTime

def getAlignedSonarFixatorPose(rawImageTime, sonarFixatorTime, sonarFixatorPose):
    '''
    根据rawImgTime的时间戳来对sonarFixatorPos进行插值，从而对齐时间戳
    :param rawImgTime:
    :param sonarFixatorTime:
    :param sonarFixatorPos:
    :return: 返回的都是与每个时段内第一张图片的时间戳的对齐之后的位姿
    '''
    SonarFixatorInterPose = np.zeros((len(rawImageTime), 4, 4))
    SonarFixatorInterPose[:, 3, -1] = 1
    # 把sonarFixatorPos60X4X4的矩阵提取出来
    for i in range(len(rawImageTime)):# 9张图片，只需要插值九次
        for j in range(3):
            for k in range(4):
                SonarFixatorInterPose[i, j, k] = np.interp(rawImageTime[i][0], sonarFixatorTime[i], np.stack(sonarFixatorPose[i], axis=0)[:, j, k])
    # aligned_sonarFixatorPos = SonarFixatorInterPose[rawImgTime]
    return SonarFixatorInterPose


def getGroundTruthMarkerPos(bag_file_path, startTime, endTime, rawImageTime):
    '''
    把marker点的准确的位置给提取出来
    rawImgTime: 是声纳图像的时间戳，传进来用来对齐Marker与声纳图像的时间戳
    '''
    # 动捕的数据单位是mm，乘以transUnit转换成m
    transUnit = 1e-3
    bag = rosbag.Bag(bag_file_path)

    # marker1捕获位姿并进行对齐
    marker1pose = []
    marker1Time = []
    for i in range(startTime.shape[0]):
        marker1Data = bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker1/pose'],
                                        start_time=rospy.Time(startTime[i]), end_time=rospy.Time(endTime[i]))
        eachMarker1 = []
        eachMarker1Time = []
        for topic, msg, t in marker1Data:
            eachMarker1.append([msg.pose.position.x * transUnit, msg.pose.position.y * transUnit, msg.pose.position.z * transUnit])
            eachMarker1Time.append(t.nsecs)
        eachMarker1 = np.stack(eachMarker1)
        eachMarker1Time = np.stack(eachMarker1Time)
        marker1pose.append(eachMarker1)
        marker1Time.append(eachMarker1Time)
        # marker1pose 和 marker1Time是包含每个时段内的位姿marker1的list,转成np.array,然后进行插值
    marker1pose = np.stack(marker1pose, axis=0)
    marker1Time = np.stack(marker1Time, axis=0)
    aligned_marker1Pose = np.zeros((len(rawImageTime), 4))
    aligned_marker1Pose[:, -1] = 1
    for i in range(startTime.shape[0]):
        for j in range(3):
            aligned_marker1Pose[i, j] = np.interp(rawImageTime[i][0], marker1Time[i], marker1pose[i, :, j])

    # marker2捕获位姿并进行对齐
    marker2pose = []
    marker2Time = []
    for i in range(startTime.shape[0]):
        marker2Data = bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker2/pose'],
                                        start_time=rospy.Time(startTime[i]), end_time=rospy.Time(endTime[i]))
        eachMarker2 = []
        eachMarker2Time = []
        for topic, msg, t in marker2Data:
            eachMarker2.append([msg.pose.position.x * transUnit, msg.pose.position.y * transUnit, msg.pose.position.z * transUnit])
            eachMarker2Time.append(t.nsecs)
        eachMarker2 = np.stack(eachMarker2)
        eachMarker2Time = np.stack(eachMarker2Time)
        marker2pose.append(eachMarker2)
        marker2Time.append(eachMarker2Time)
        # marker1pose 和 marker1Time是包含每个时段内的位姿marker1的list,转成np.array,然后进行插值
    marker2pose = np.stack(marker2pose, axis=0)
    marker2Time = np.stack(marker2Time, axis=0)
    aligned_marker2Pose = np.zeros((len(rawImageTime), 4))
    aligned_marker2Pose[:, -1] = 1
    for i in range(startTime.shape[0]):
        for j in range(3):
            aligned_marker2Pose[i, j] = np.interp(rawImageTime[i][0], marker2Time[i], marker2pose[i, :, j])

    # marker3捕获位姿并进行对齐
    marker3pose = []
    marker3Time = []
    for i in range(startTime.shape[0]):
        marker3Data = bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker3/pose'],
                                        start_time=rospy.Time(startTime[i]), end_time=rospy.Time(endTime[i]))
        eachMarker3 = []
        eachMarker3Time = []
        for topic, msg, t in marker3Data:
            eachMarker3.append(
                [msg.pose.position.x * transUnit, msg.pose.position.y * transUnit, msg.pose.position.z * transUnit])
            eachMarker3Time.append(t.nsecs)
        eachMarker3 = np.stack(eachMarker3)
        eachMarker3Time = np.stack(eachMarker3Time)
        marker3pose.append(eachMarker3)
        marker3Time.append(eachMarker3Time)
        # marker1pose 和 marker1Time是包含每个时段内的位姿marker1的list,转成np.array,然后进行插值
    marker3pose = np.stack(marker3pose, axis=0)
    marker3Time = np.stack(marker3Time, axis=0)
    aligned_marker3Pose = np.zeros((len(rawImageTime), 4))
    aligned_marker3Pose[:, -1] = 1
    for i in range(startTime.shape[0]):
        for j in range(3):
            aligned_marker3Pose[i, j] = np.interp(rawImageTime[i][0], marker3Time[i], marker3pose[i, :, j])

    # marker4捕获位姿并进行对齐
    marker4pose = []
    marker4Time = []
    for i in range(startTime.shape[0]):
        marker4Data = bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker4/pose'],
                                        start_time=rospy.Time(startTime[i]), end_time=rospy.Time(endTime[i]))
        eachMarker4 = []
        eachMarker4Time = []
        for topic, msg, t in marker4Data:
            eachMarker4.append(
                [msg.pose.position.x * transUnit, msg.pose.position.y * transUnit, msg.pose.position.z * transUnit])
            eachMarker4Time.append(t.nsecs)
        eachMarker4 = np.stack(eachMarker4)
        eachMarker4Time = np.stack(eachMarker4Time)
        marker4pose.append(eachMarker4)
        marker4Time.append(eachMarker4Time)
        # marker1pose 和 marker1Time是包含每个时段内的位姿marker1的list,转成np.array,然后进行插值
    marker4pose = np.stack(marker4pose, axis=0)
    marker4Time = np.stack(marker4Time, axis=0)
    aligned_marker4Pose = np.zeros((len(rawImageTime), 4))
    aligned_marker4Pose[:, -1] = 1
    for i in range(startTime.shape[0]):
        for j in range(3):
            aligned_marker4Pose[i, j] = np.interp(rawImageTime[i][0], marker4Time[i], marker4pose[i, :, j])

    # 返回时间戳已经对齐了的marker
    return aligned_marker1Pose, aligned_marker2Pose, aligned_marker3Pose, aligned_marker4Pose

if __name__ == '__main__':
    # startTime是所有可用的图片的, 因为第7张声纳图像和第8张声纳图像有问题，所以把他给删掉
    startTime = np.array(
        [1705404798.0, 1705404799.0, 1705404800.0, 1705404801.0, 1705404803.0, 1705404805.0, 1705404842.0, 1705404844.0, 1705404851.0, 1705404853.0, 1705404854.0, 1705404862.0])
    endTime = startTime[:, ] + 1
    bag_file_path = '/home/titanium/ubuntu/sonarCalibration/final/final_biaoding/first_bag/2024-01-16-19-33-16.bag'
    # 这里的imgTime要取到每个image时间内的第一帧图像的时间戳
    save_image_path = '/home/titanium/ubuntu/sonarCalibration/final/final_biaoding/first_bag/image'
    imageTime = save_image(bag_file_path, startTime, endTime, save_image_path)
    save_raw_image_path = '/home/titanium/ubuntu/sonarCalibration/final/final_biaoding/first_bag/rawImage'
    rawImageTime = save_raw_image(bag_file_path, startTime, endTime, save_raw_image_path)

    # 获得SonarFixator动捕位姿及其时间戳
    sonarFixatorPose, sonarFixatorTime = getSonarFixatorPose(bag_file_path, startTime, endTime)
    # 对齐SonarFixator动捕的时间戳
    aligned_sonarFixatorPose = getAlignedSonarFixatorPose(rawImageTime, sonarFixatorTime, sonarFixatorPose)

    # 获得对齐时间戳的marker点的准确的三维信息, GroundTruth的数值单位是m
    marker1GT, marker2GT, marker3GT, marker4GT = getGroundTruthMarkerPos(bag_file_path, startTime, endTime, rawImageTime)

    #将动捕的信息保存下来
    motionCaptureData = {'marker1GT': marker1GT, 'marker2GT': marker2GT, 'marker3GT' : marker3GT, 'marker4GT' : marker4GT,
                         'aligned_sonarFixatorPose' : aligned_sonarFixatorPose}
    with open('./motionCaptureData.pkl', 'wb') as file:
        pickle.dump(motionCaptureData, file)

