import os
import numpy as np
import cv2
import rosbag
import pickle
import rospy
from forPlot import angle_with_z_axis
import matplotlib.pyplot as plt
# from denoise import denoise

def getMarkerCoordInSonar(TsInW, aligned_marker1_x, aligned_marker1_y, aligned_marker1_z, aligned_marker2_x, aligned_marker2_y, aligned_marker2_z, aligned_marker3_x, aligned_marker3_y, aligned_marker3_z, aligned_marker4_x, aligned_marker4_y, aligned_marker4_z):
    TwInS = np.linalg.inv(TsInW)
    marker1InS = TwInS @ np.array([aligned_marker1_x, aligned_marker1_y, aligned_marker1_z, 1])
    marker2InS = TwInS @ np.array([aligned_marker2_x, aligned_marker2_y, aligned_marker2_z, 1])
    marker3InS = TwInS @ np.array([aligned_marker3_x, aligned_marker3_y, aligned_marker3_z, 1])
    marker4InS = TwInS @ np.array([aligned_marker4_x, aligned_marker4_y, aligned_marker4_z, 1])

    return marker1InS, marker2InS, marker3InS, marker4InS

def saveBackgroungNoise():
    # 先读取一下带噪声的bag
    noiseBackground_filePath = './20240110data/beijing.bag'
    saveDir = './20240110data/noise'
    noiseBag = rosbag.Bag(noiseBackground_filePath)
    i = 0
    for topic, msg, t in noiseBag.read_messages(topics=['/sonar_oculus_node/raw_img']):
        img = np.frombuffer(msg.data, np.uint8)
        img = img.reshape((msg.height, msg.width))
        ImagingSonar = np.asarray(img / 255., dtype=np.float32)
        data = {"ImagingSonar": ImagingSonar}
        with open(f'{saveDir}/{i}.pkl', 'wb') as file:
            pickle.dump(data, file, protocol=3)
            i += 1



def getTransMatrix(aligned_p_x, aligned_p_y, aligned_p_z, aligned_q_x, aligned_q_y, aligned_q_z, aligned_q_w):
    '''

    :param aligned_p_x:
    :param aligned_p_y:
    :param aligned_p_z:
    :param aligned_q_x:
    :param aligned_q_y:
    :param aligned_q_z:
    :param aligned_q_w:
    :return:返回一个Body的变换矩阵
    '''
    p_x, p_y, p_z, q_x, q_y, q_z, q_w = aligned_p_x, aligned_p_y, aligned_p_z, aligned_q_x, aligned_q_y, aligned_q_z, aligned_q_w
    transMatrix = np.eye(4, 4)
    transMatrix[:3, :3] = np.array([
        [1 - 2 * (q_y ** 2 + q_z ** 2), 2 * (q_x * q_y - q_z * q_w), 2 * (q_x * q_z + q_y * q_w)],
        [2 * (q_x * q_y + q_z * q_w), 1 - 2 * (q_x ** 2 + q_z ** 2), 2 * (q_y * q_z - q_x * q_w)],
        [2 * (q_x * q_z - q_y * q_w), 2 * (q_y * q_z + q_x * q_w), 1 - 2 * (q_x ** 2 + q_y ** 2)]
    ])
    transMatrix[:3, -1] = np.array([p_x, p_y, p_z])
    return transMatrix

def save_raw_image(bag_file_path, saveDir, num, delta_t):
    # bag_file_path = '/home/titanium/ubuntu/underWaterData/Chair/ChairNew/Chair/2023-12-20-00-07-04.bag'
    # 声纳标定计算出来的变换矩阵
    TbInS = np.array([[0.7232646, -0.00657526, -0.68748181, 0.02618163],
                      [-0.52574364, 0.6418269, -0.55335156, -0.09792483],
                      [0.44302155, 0.7632569, 0.46148769, 0.04740003],
                      [0., 0., 0., 1.]])
    TsInB = np.linalg.inv(TbInS)
    index = num
    bag = rosbag.Bag(bag_file_path)
    # 把rawImage保存下来
    rawImageTime = []
    img_data = []
    raw_image_data = bag.read_messages(topics=['/sonar_oculus_node/raw_img'])
    for topic, msg, t in raw_image_data:

        startTimeForDetectPose = rospy.Time.from_sec(t.to_sec() - delta_t)
        endTimeForDetectPose = rospy.Time.from_sec(t.to_sec() + delta_t)

        # 确保在当前的声纳图像附近存在marker点的动捕坐标和body的动捕位姿
        # 获取在startTimeForDetectPose ～ endTimeForDetectPose时间范围内'/vrpn_client_node/TrackerSonar/pose'话题的消息数量
        sonarPoseMsgTag = False
        for _, _, temp_t in bag.read_messages(topics=['/vrpn_client_node/TrackerSonar/pose'], start_time=startTimeForDetectPose, end_time=endTimeForDetectPose):
            sonarPoseMsgTag = True
            break
        marker1PoseMsgTag = False
        for _, _, temp_t in bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker1/pose'], start_time=startTimeForDetectPose, end_time=endTimeForDetectPose):
            marker1PoseMsgTag = True
            break
        marker2PoseMsgTag = False
        for _, _, temp_t in bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker2/pose'],
                                              start_time=startTimeForDetectPose, end_time=endTimeForDetectPose):
            marker2PoseMsgTag = True
            break
        marker3PoseMsgTag = False
        for _, _, temp_t in bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker3/pose'],
                                              start_time=startTimeForDetectPose, end_time=endTimeForDetectPose):
            marker3PoseMsgTag = True
            break
        marker4PoseMsgTag = False
        for _, _, temp_t in bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker4/pose'],
                                              start_time=startTimeForDetectPose, end_time=endTimeForDetectPose):
            marker4PoseMsgTag = True
            break

        if sonarPoseMsgTag & marker1PoseMsgTag & marker2PoseMsgTag & marker3PoseMsgTag & marker4PoseMsgTag:
            # 只有动捕消息都存在时才将图像数据保存下来
            rawImageTime.append(t.to_sec())
            img = np.frombuffer(msg.data, np.uint8)
            img = img.reshape((msg.height, msg.width))
            ImagingSonar = np.asarray(img / 255., dtype=np.float32)
            img_data.append(ImagingSonar)
        else:
            continue

        # cv2.imshow('raw_image', img)
        # cv2.waitKey(0)

    rawImageTime = np.stack(rawImageTime) # 先获取一个时间戳，然后
    # 获取的位姿数据是声纳上面的反光球组成的刚体的位姿
    sonar_pose_data = bag.read_messages(topics='/vrpn_client_node/TrackerSonar/pose')
    p_x, p_y, p_z, q_x, q_y, q_z, q_w = [], [], [], [], [], [], []
    pose_time = []
    for topic, msg, t in sonar_pose_data:
        p_x.append(msg.pose.position.x)
        p_y.append(msg.pose.position.y)
        p_z.append(msg.pose.position.z)
        q_x.append(msg.pose.orientation.x)
        q_y.append(msg.pose.orientation.y)
        q_z.append(msg.pose.orientation.z)
        q_w.append(msg.pose.orientation.w)
        pose_time.append(t.to_sec())
    p_x, p_y, p_z, q_x, q_y, q_z, q_w = np.stack(p_x), np.stack(p_y), np.stack(p_z), np.stack(q_x), np.stack(q_y), np.stack(q_z), np.stack(q_w)
    # 开始插值
    aligned_p_x = np.interp(rawImageTime, pose_time, p_x)
    aligned_p_y = np.interp(rawImageTime, pose_time, p_y)
    aligned_p_z = np.interp(rawImageTime, pose_time, p_z)
    aligned_q_x = np.interp(rawImageTime, pose_time, q_x)
    aligned_q_y = np.interp(rawImageTime, pose_time, q_y)
    aligned_q_z = np.interp(rawImageTime, pose_time, q_z)
    aligned_q_w = np.interp(rawImageTime, pose_time, q_w)
    # 进行单位变换
    transUnit = 1e-3

    # 获得对齐时间戳的marker点的动捕数据
    marker1_data = bag.read_messages(topics='/vrpn_client_node/TrackerBoard_Marker1/pose')
    marker1_x, marker1_y, marker1_z = [], [], []
    marker1_time = []
    for topic, msg, t in marker1_data:
        marker1_x.append(msg.pose.position.x * transUnit)
        marker1_y.append(msg.pose.position.y * transUnit)
        marker1_z.append(msg.pose.position.z * transUnit)
        marker1_time.append(t.to_sec())
    marker1_x, marker1_y, marker1_z = np.stack(marker1_x), np.stack(marker1_y), np.stack(marker1_z)
    # marker1的时间戳对齐rawImage
    aligned_marker1_x = np.interp(rawImageTime, marker1_time, marker1_x)
    aligned_marker1_y = np.interp(rawImageTime, marker1_time, marker1_y)
    aligned_marker1_z = np.interp(rawImageTime, marker1_time, marker1_z)

    marker2_data = bag.read_messages(topics='/vrpn_client_node/TrackerBoard_Marker2/pose')
    marker2_x, marker2_y, marker2_z = [], [], []
    marker2_time = []
    for topic, msg, t in marker2_data:
        marker2_x.append(msg.pose.position.x * transUnit)
        marker2_y.append(msg.pose.position.y * transUnit)
        marker2_z.append(msg.pose.position.z * transUnit)
        marker2_time.append(t.to_sec())
    marker2_x, marker2_y, marker2_z = np.stack(marker2_x), np.stack(marker2_y), np.stack(marker2_z)
    # marker1的时间戳对齐rawImage
    aligned_marker2_x = np.interp(rawImageTime, marker2_time, marker2_x)
    aligned_marker2_y = np.interp(rawImageTime, marker2_time, marker2_y)
    aligned_marker2_z = np.interp(rawImageTime, marker2_time, marker2_z)

    marker3_data = bag.read_messages(topics='/vrpn_client_node/TrackerBoard_Marker3/pose')
    marker3_x, marker3_y, marker3_z = [], [], []
    marker3_time = []
    for topic, msg, t in marker3_data:
        marker3_x.append(msg.pose.position.x * transUnit)
        marker3_y.append(msg.pose.position.y * transUnit)
        marker3_z.append(msg.pose.position.z * transUnit)
        marker3_time.append(t.to_sec())
    marker3_x, marker3_y, marker3_z = np.stack(marker3_x), np.stack(marker3_y), np.stack(marker3_z)
    # marker1的时间戳对齐rawImage
    aligned_marker3_x = np.interp(rawImageTime, marker3_time, marker3_x)
    aligned_marker3_y = np.interp(rawImageTime, marker3_time, marker3_y)
    aligned_marker3_z = np.interp(rawImageTime, marker3_time, marker3_z)

    marker4_data = bag.read_messages(topics='/vrpn_client_node/TrackerBoard_Marker4/pose')
    marker4_x, marker4_y, marker4_z = [], [], []
    marker4_time = []
    for topic, msg, t in marker4_data:
        marker4_x.append(msg.pose.position.x * transUnit)
        marker4_y.append(msg.pose.position.y * transUnit)
        marker4_z.append(msg.pose.position.z * transUnit)
        marker4_time.append(t.to_sec())
    marker4_x, marker4_y, marker4_z = np.stack(marker4_x), np.stack(marker4_y), np.stack(marker4_z)
    # marker1的时间戳对齐rawImage
    aligned_marker4_x = np.interp(rawImageTime, marker4_time, marker4_x)
    aligned_marker4_y = np.interp(rawImageTime, marker4_time, marker4_y)
    aligned_marker4_z = np.interp(rawImageTime, marker4_time, marker4_z)

    height = 707
    width = 512
    max_Range = 2.  # 单位m
    horizontal_Angle = 60.  # 这里的度数是角度制
    r_resolution = max_Range / height
    theta_resolution = horizontal_Angle / width
    thickness = 2
    radius = 2
    color = (255, 255, 255)  # 使用BGR格式，这里是绿色,重投影的用蓝色
    # 把动捕搜集的marker点变换到声纳图像下
    for i in range(rawImageTime.shape[0]):
        TbInW = getTransMatrix(aligned_p_x[i] * transUnit, aligned_p_y[i] * transUnit, aligned_p_z[i] * transUnit, aligned_q_x[i], aligned_q_y[i], aligned_q_z[i], aligned_q_w[i])
        # 上一行的poseSensor是Body系下的，获得声纳在世界坐标系下的位姿然后保存下来
        TsInW = TbInW @ TsInB
        # 利用TsInW计算与Z轴的夹角
        sonar_z_vector = TsInW[:3, 2]
        angle = angle_with_z_axis(sonar_z_vector)
        ImagingSonar = img_data[i]
        ImagingSonar = np.round(ImagingSonar * 255.0).astype(np.uint8)
        ImagingSonar = cv2.applyColorMap(ImagingSonar, cv2.COLORMAP_JET)
        # 把世界坐标系下的动捕点转换到图像坐标系下, 并且绘图
        marker1InS, marker2InS, marker3InS, marker4InS = getMarkerCoordInSonar(TsInW, aligned_marker1_x[i], aligned_marker1_y[i], aligned_marker1_z[i], aligned_marker2_x[i], aligned_marker2_y[i], aligned_marker2_z[i], aligned_marker3_x[i], aligned_marker3_y[i], aligned_marker3_z[i], aligned_marker4_x[i], aligned_marker4_y[i], aligned_marker4_z[i])
        # 绘制marker1
        marker1PolarCoord = np.array([np.sqrt(marker1InS[0] ** 2 + marker1InS[1] ** 2 + marker1InS[2] ** 2),
                                      np.degrees(np.mod(np.arctan(marker1InS[1] / marker1InS[0]), np.pi))])  # (r, theta)
        myMarker1Coord = np.array([int(marker1PolarCoord[0] / r_resolution),
                                   int((90 - marker1PolarCoord[1]) / theta_resolution + width / 2)])
        cv2.circle(ImagingSonar, (int(myMarker1Coord[1]), int(myMarker1Coord[0])), radius, color, thickness)

        # 绘制marker2
        marker2PolarCoord = np.array([np.sqrt(marker2InS[0] ** 2 + marker2InS[1] ** 2 + marker2InS[2] ** 2),
                                      np.degrees(
                                          np.mod(np.arctan(marker2InS[1] / marker2InS[0]), np.pi))])  # (r, theta)
        myMarker2Coord = np.array([int(marker2PolarCoord[0] / r_resolution),
                                   int((90 - marker2PolarCoord[1]) / theta_resolution + width / 2)])
        cv2.circle(ImagingSonar, (int(myMarker2Coord[1]), int(myMarker2Coord[0])), radius, color, thickness)

        # 绘制marker3
        marker3PolarCoord = np.array([np.sqrt(marker3InS[0] ** 2 + marker3InS[1] ** 2 + marker3InS[2] ** 2),
                                      np.degrees(
                                          np.mod(np.arctan(marker3InS[1] / marker3InS[0]), np.pi))])  # (r, theta)
        myMarker3Coord = np.array([int(marker3PolarCoord[0] / r_resolution),
                                   int((90 - marker3PolarCoord[1]) / theta_resolution + width / 2)])
        cv2.circle(ImagingSonar, (int(myMarker3Coord[1]), int(myMarker3Coord[0])), radius, color, thickness)

        # 绘制marker4
        marker4PolarCoord = np.array([np.sqrt(marker4InS[0] ** 2 + marker4InS[1] ** 2 + marker4InS[2] ** 2),
                                      np.degrees(
                                          np.mod(np.arctan(marker4InS[1] / marker4InS[0]), np.pi))])  # (r, theta)
        myMarker4Coord = np.array([int(marker4PolarCoord[0] / r_resolution),
                                   int((90 - marker4PolarCoord[1]) / theta_resolution + width / 2)])
        cv2.circle(ImagingSonar, (int(myMarker4Coord[1]), int(myMarker4Coord[0])), radius, color, thickness)

        # 保存图像
        cv2.imwrite(f"{saveDir}/{i + index}-{angle}.png", ImagingSonar)
        # plt.imsave(f"{saveDir}/{i + index}.png", ImagingSonar, cmap='gray')
        # plt.savefig(f"{saveDir}/{i + index}.png")
        num += 1
    bag.close()
    return num

if __name__ == '__main__':
    # saveBackgroungNoise()
    # 标定板的bag数据路径
    fileDir = '/home/titanium/ubuntu/sonarCalibration/final/final_biaoding/bags'
    # 第一次保存重投影的声纳图像的示例
    # saveDir = '/home/titanium/ubuntu/sonarCalibration/1228Board/reProjectionImage'
    # 第二次保存，把计算的z轴方向的角度保存到文件名中
    saveDir = '/home/titanium/ubuntu/sonarCalibration/final/final_biaoding/reProjectionImage'
    num = 0 # 用于记录数目
    delta_t = 0.1 # 如果一张声纳图像前后delta_t时间内有动捕位姿，就把这个声纳图像保存下来
    with os.scandir(fileDir) as entries:
        for bag in entries:
            print("正在处理{}...".format(bag.name))
            num = save_raw_image(bag.path, saveDir, num, delta_t)
            print("{}bag 的{}条数据已经保存成功...".format(bag.name, num))