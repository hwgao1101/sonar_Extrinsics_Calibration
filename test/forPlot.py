import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import rosbag

def angle_with_z_axis(vector):
    # 定义 Z 轴方向的单位向量
    z_axis = np.array([0, 0, 1])

    # 计算向量与 Z 轴方向的点积
    dot_product = np.dot(vector, z_axis)

    # 计算向量的模和 Z 轴方向的模
    vector_magnitude = np.linalg.norm(vector)
    z_axis_magnitude = np.linalg.norm(z_axis)

    # 计算夹角（弧度）
    angle_radians = np.arccos(dot_product / (vector_magnitude * z_axis_magnitude))

    # 将弧度转换为角度
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

# 示例：计算一个向量与 Z 轴方向的夹角
vector = np.array([1, 1, 2])
angle = angle_with_z_axis(vector)

def plotSonarFixatorPose(aligned_sonarFixatorPose):
    # 创建3D图形和坐标轴
    # 定义原始
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    original_axes = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
    ax.quiver(0, 0, 0, original_axes[0, 0], original_axes[1, 0], original_axes[2, 0], color='red', label='X Axis', length=0.5)
    ax.quiver(0, 0, 0, original_axes[0, 1], original_axes[1, 1], original_axes[2, 1], color='green', label='Y Axis', length=0.5)
    ax.quiver(0, 0, 0, original_axes[0, 2], original_axes[1, 2], original_axes[2, 2], color='blue', label='Z Axis', length=0.5)
    for i in range(aligned_sonarFixatorPose.shape[0]):
        axis_length = 0.2
        # 提取旋转矩阵中的三个轴
        x_axis = aligned_sonarFixatorPose[i, :3, 0] * axis_length
        y_axis = aligned_sonarFixatorPose[i, :3, 1] * axis_length
        z_axis = aligned_sonarFixatorPose[i, :3, 2] * axis_length
        # 绘制变换后的坐标轴
        ax.quiver(*aligned_sonarFixatorPose[i, :3, -1], *x_axis, color='r', label='X Axis', length=axis_length)
        ax.quiver(*aligned_sonarFixatorPose[i, :3, -1], *y_axis, color='g', label='Y Axis', length=axis_length)
        ax.quiver(*aligned_sonarFixatorPose[i, :3, -1], *z_axis, color='b', label='Z Axis', length=axis_length) # , linestyle='dashed'

        TbInS = np.array([[ 0.7232646 , -0.00657526, -0.68748181,  0.02618163],
                            [-0.52574364,  0.6418269 , -0.55335156, -0.09792483],
                            [ 0.44302155,  0.7632569 ,  0.46148769,  0.04740003],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]])
        TsInW = aligned_sonarFixatorPose[i] @ np.linalg.inv(TbInS)
        # 计算z轴方向夹角
        sonar_z_vector = TsInW[:3, 2]
        angle = angle_with_z_axis(sonar_z_vector)

        xx_axis = TsInW[:3, 0] * axis_length
        yy_axis = TsInW[:3, 1] * axis_length
        zz_axis = TsInW[:3, 2] * axis_length
        ax.quiver(*TsInW[:3, -1], *xx_axis, color='r', label=f'{i}-{angle}-X Axis:{TsInW[0, -1]}', length=axis_length, linestyle='dashed')
        ax.quiver(*TsInW[:3, -1], *yy_axis, color='g', label=f'{i}-{angle}-Y Axis:{TsInW[1, -1]}', length=axis_length, linestyle='dashed') # label='Y Axis',
        ax.quiver(*TsInW[:3, -1], *zz_axis, color='b', label=f'{i}-{angle}-Z Axis:{TsInW[2, -1]}', length=axis_length, linestyle='dashed') # label='Z Axis',

        ax.text(TsInW[0, -1], TsInW[1, -1], TsInW[2, -1], f'{i}', color='k', fontsize=12)

    # 显示图例
    ax.legend()
    # 显示图形
    plt.show()
def plotMotionCaptureForMarker(bag_file_path):
    bag = rosbag.Bag(bag_file_path)
    #
    marker1X, marker1Y, marker1Z, marker1Time = [], [], [], []
    marker2X, marker2Y, marker2Z, marker2Time = [], [], [], []
    marker3X, marker3Y, marker3Z, marker3Time = [], [], [], []
    marker4X, marker4Y, marker4Z, marker4Time = [], [], [], []
    marker1_data = bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker1/pose'])
    for topic, msg, t in marker1_data:
        marker1Time.append(t.to_sec())
        marker1X.append(msg.pose.position.x)
        marker1Y.append(msg.pose.position.y)
        marker1Z.append(msg.pose.position.z)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
    ax1.plot(marker1Time, marker1X, label='marker1 - X')
    ax1.set_title('marker1 - X')
    ax1.legend()
    ax2.plot(marker1Time, marker1Y, label='marker1 - Y')
    ax2.set_title('marker1 - Y')
    ax3.legend()
    ax3.plot(marker1Time, marker1Z, label='marker1 - Z')
    ax2.set_title('marker1 - Z')
    ax3.legend()
    plt.show()
    # marker2_data = bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker2/pose'])
    # for topic, msg, t in marker2_data:
    #     marker2Time.append(t.to_sec())
    #     marker2X.append(msg.pose.x)
    #     marker2Y.append(msg.pose.y)
    #     marker2Z.append(msg.pose.z)
    # marker3_data = bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker3/pose'])
    # for topic, msg, t in marker3_data:
    #     marker3Time.append(t.to_sec())
    #     marker3X.append(msg.pose.x)
    #     marker3Y.append(msg.pose.y)
    #     marker3Z.append(msg.pose.z)
    # marker4_data = bag.read_messages(topics=['/vrpn_client_node/TrackerBoard_Marker4/pose'])
    # for topic, msg, t in marker3_data:
    #     marker4Time.append(t.to_sec())
    #     marker4X.append(msg.pose.x)
    #     marker4Y.append(msg.pose.y)
    #     marker4Z.append(msg.pose.z)


if __name__ == '__main__':
    # 绘制动捕对于marker的位姿曲线
    bag_file_path = '/home/titanium/ubuntu/sonarCalibration/final/final_biaoding/first_bag/2024-01-16-19-33-16.bag'
    plotMotionCaptureForMarker(bag_file_path)


    with open('motionCaptureData.pkl', 'rb') as motionFile:
        motionCaptureData = pickle.load(motionFile)
    marker1GT, marker2GT, marker3GT, marker4GT = motionCaptureData['marker1GT'], motionCaptureData['marker2GT'], motionCaptureData['marker3GT'], motionCaptureData['marker4GT']
    aligned_sonarFixatorPose = motionCaptureData['aligned_sonarFixatorPose']
    # 把sonarFixatorPose在世界坐标系中的位姿绘制出来
    plotSonarFixatorPose(aligned_sonarFixatorPose)
    marker1InBody, marker2InBody, marker3InBody, marker4InBody = np.zeros((12, 4)), np.zeros((12, 4)), np.zeros((12, 4)), np.zeros((12, 4))
    for i in range(aligned_sonarFixatorPose.shape[0]):
        marker1InBody[i] = aligned_sonarFixatorPose[i] @ marker1GT[i]
        marker2InBody[i] = aligned_sonarFixatorPose[i] @ marker2GT[i]
        marker3InBody[i] = aligned_sonarFixatorPose[i] @ marker3GT[i]
        marker4InBody[i] = aligned_sonarFixatorPose[i] @ marker4GT[i]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(marker1InBody[:, 0], marker1InBody[:, 1], marker1InBody[:, 2], c='r', marker='o')
    # ax.scatter(marker2InBody[:, 0], marker2InBody[:, 1], marker2InBody[:, 2], c='b', marker='o')
    # ax.scatter(marker3InBody[:, 0], marker3InBody[:, 1], marker3InBody[:, 2], c='y', marker='o')
    # ax.scatter(marker4InBody[:, 0], marker4InBody[:, 1], marker4InBody[:, 2], c='g', marker='o')
    # # 设置坐标轴标签
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    for i in range(marker1InBody.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(marker1InBody[i, 0], marker1InBody[i, 1], marker1InBody[i, 2], c='r', marker='o', s=30)
        ax.scatter(marker2InBody[i, 0], marker2InBody[i, 1], marker2InBody[i, 2], c='b', marker='o', s=30)
        ax.scatter(marker3InBody[i, 0], marker3InBody[i, 1], marker3InBody[i, 2], c='y', marker='o', s=30)
        ax.scatter(marker4InBody[i, 0], marker4InBody[i, 1], marker4InBody[i, 2], c='g', marker='o', s=30)
        # 绘制坐标轴出来
        ax.quiver(0, 0, 0, 0.1, 0, 0, color='red', label='X Axis')
        ax.quiver(0, 0, 0, 0, 0.1, 0, color='green', label='Y Axis')
        ax.quiver(0, 0, 0, 0, 0, 0.1, color='blue', label='Z Axis')
        # 设置坐标轴标签
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # 显示图形
        # plt.show()
    #     plt.savefig(f'./test/{i}.png')
    print("finished!")