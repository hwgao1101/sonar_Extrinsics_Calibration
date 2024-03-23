import pickle
import numpy as np
from scipy.optimize import least_squares

'''
多张声纳图像进行标定
'''

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])

def fromRotationGetquaternion(R):
    trace_R = np.trace(R)
    w = np.sqrt(1 + trace_R) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])

def getInitialRotation(yaw, pith, roll):
    # R_z = np.eye(3)
    # R_y = np.eye(3)
    # R_x = np.eye(3)
    #
    # R_z[0, 0] = np.cos(np.radians(yaw))
    # R_z[0, 1] = -np.sin(np.radians(yaw))
    # R_z[1, 0] = np.sin(np.radians(yaw))
    # R_z[1, 1] = np.cos(np.radians(yaw))
    quaternion_z = np.array([np.cos(np.radians(yaw) / 2), 0, 0, np.sin(np.radians(yaw) / 2)])

    # R_y[0, 0] = np.cos(np.radians(pith))
    # R_y[0, 2] = np.sin(np.radians(pith))
    # R_y[2, 0] = -np.sin(np.radians(pith))
    # R_y[2, 2] = np.cos(np.radians(pith))
    quaternion_y = np.array([np.cos(np.radians(pith) / 2), 0, np.sin(np.radians(pith) / 2), 0])

    # R_x[1, 1] = np.cos(np.radians(roll))
    # R_x[1, 2] = -np.sin(np.radians(roll))
    # R_x[2, 1] = np.sin(np.radians(roll))
    # R_x[2, 2] = np.cos(np.radians(roll))
    quaternion_x = np.array([np.cos(np.radians(roll) /2 ), np.sin(np.radians(roll) /2 ), 0, 0])

    return quaternion_multiply(quaternion_z, quaternion_multiply(quaternion_y, quaternion_x))

def multiCaliEquations(vars, marker1, marker2, marker3, marker4, marker1GT, marker2GT, marker3GT, marker4GT, aligned_sonarFixatorPose):

    # 待求变量设置
    q_w, q_x, q_y, q_z, p_x, p_y, p_z = vars[0: 7]
    # phi1 = vars[7: 16] # 9个
    # phi2 = vars[16: 25]
    # phi3 = vars[25: 34]
    # phi4 = vars[34: 43]

    phi1 = vars[7: 19]  # 7个
    phi2 = vars[19: 31]
    phi3 = vars[31: 43]
    phi4 = vars[43: 55]

    # 额外常数设置
    XaInW, XbInW, XcInW, XdInW, XfInW = marker1GT, marker2GT, marker3GT, marker4GT, aligned_sonarFixatorPose

    # 获得 XaInS
    # XaInS = np.ones((9, 4))
    XaInS = np.ones((12, 4))
    # marker二维坐标中存的水平偏转角是度数制的，使用np.radians()转成弧度制
    XaInS[:, 0] = marker1[:, 0] * np.cos(np.radians(phi1)) * np.cos(np.radians(marker1[:, 1]))# 传进来的marker1[0]是距离，单位m, marker1[1]是角度，单位为度
    XaInS[:, 1] = marker1[:, 0] * np.cos(np.radians(phi1)) * np.sin(np.radians(marker1[:, 1]))
    XaInS[:, 2] = marker1[:, 0] * np.sin(np.radians(phi1))
    # 获得 XbInS
    XbInS = np.ones((12, 4))
    XbInS[:, 0] = marker2[:, 0] * np.cos(np.radians(phi2)) * np.cos(np.radians(marker2[:, 1]))
    XbInS[:, 1] = marker2[:, 0] * np.cos(np.radians(phi2)) * np.sin(np.radians(marker2[:, 1]))
    XbInS[:, 2] = marker2[:, 0] * np.sin(np.radians(phi2))
    # 获得 XcInS
    XcInS = np.ones((12, 4))
    XcInS[:, 0] = marker3[:, 0] * np.cos(np.radians(phi3)) * np.cos(np.radians(marker3[:, 1]))
    XcInS[:, 1] = marker3[:, 0] * np.cos(np.radians(phi3)) * np.sin(np.radians(marker3[:, 1]))
    XcInS[:, 2] = marker3[:, 0] * np.sin(np.radians(phi3))
    # 获得 XdInS
    XdInS = np.ones((12, 4))
    XdInS[:, 0] = marker4[:, 0] * np.cos(np.radians(phi4)) * np.cos(np.radians(marker4[:, 1]))
    XdInS[:, 1] = marker4[:, 0] * np.cos(np.radians(phi4)) * np.sin(np.radians(marker4[:, 1]))
    XdInS[:, 2] = marker4[:, 0] * np.sin(np.radians(phi4))

    # 从Fixator向声纳的变换矩阵
    Tf2s = np.eye(4, 4)
    # 填进旋转矩阵
    Tf2s[:3, :3] = np.array([
        [1 - 2 * (q_y ** 2 + q_z ** 2), 2 * (q_x * q_y - q_z * q_w), 2 * (q_x * q_z + q_y * q_w)],
        [2 * (q_x * q_y + q_z * q_w), 1 - 2 * (q_x ** 2 + q_z ** 2), 2 * (q_y * q_z - q_x * q_w)],
        [2 * (q_x * q_z - q_y * q_w), 2 * (q_y * q_z + q_x * q_w), 1 - 2 * (q_x ** 2 + q_y ** 2)]
    ])
    # 填进平移矩阵
    Tf2s[:3, -1] = np.array([p_x, p_y, p_z])

    # 列写方程
    eq1 = np.zeros((12, 3))
    eq2 = np.zeros((12, 3))
    eq3 = np.zeros((12, 3))
    eq4 = np.zeros((12, 3))
    for i in range(12):
        eq1[i] = (Tf2s @ np.linalg.inv(XfInW[i]) @ XaInW[i] - XaInS[i])[:3]
        eq2[i] = (Tf2s @ np.linalg.inv(XfInW[i]) @ XbInW[i] - XbInS[i])[:3]
        eq3[i] = (Tf2s @ np.linalg.inv(XfInW[i]) @ XcInW[i] - XcInS[i])[:3]
        eq4[i] = (Tf2s @ np.linalg.inv(XfInW[i]) @ XdInW[i] - XdInS[i])[:3]
    eq5 = np.array(np.sqrt(q_x**2 + q_y**2 + q_z**2 + q_w**2) - 1.)
    return np.append((np.concatenate([eq1, eq2, eq3, eq4])).reshape(144,), eq5)
if __name__ == '__main__':
    # 加载数据进来之火把有问题的第7个、第8个给删掉
    # indices_to_delete = np.array([6, 7])
    # 加载pkl数据进来
    with open('markerCoordData.pkl', 'rb') as coordFile:
        markerCoordData = pickle.load(coordFile)
    marker1, marker2, marker3, marker4 = markerCoordData['marker1'], markerCoordData['marker2'], markerCoordData['marker3'], markerCoordData['marker4']
    # marker1, marker2, marker3, marker4 = np.delete(marker1, indices_to_delete, axis=0), np.delete(marker2, indices_to_delete, axis=0), np.delete(marker3, indices_to_delete, axis=0), np.delete(marker4, indices_to_delete, axis=0)
    with open('motionCaptureData.pkl', 'rb') as motionFile:
        motionCaptureData = pickle.load(motionFile)

    marker1GT, marker2GT, marker3GT, marker4GT = motionCaptureData['marker1GT'], motionCaptureData['marker2GT'], motionCaptureData['marker3GT'], motionCaptureData['marker4GT']
    # marker1GT, marker2GT, marker3GT, marker4GT = np.delete(marker1GT, indices_to_delete, axis=0), np.delete(marker2GT, indices_to_delete, axis=0), np.delete(marker3GT, indices_to_delete, axis=0), np.delete(marker4GT, indices_to_delete, axis=0)
    aligned_sonarFixatorPose = motionCaptureData['aligned_sonarFixatorPose']
    # aligned_sonarFixatorPose = np.delete(aligned_sonarFixatorPose, indices_to_delete, axis=0)
    print('load data finished!')

    # 进行标定
    # initial_value = np.zeros(43) # 待求的未知数个数43： 一个变换矩阵4+3个未知数， 其余每张图片4个抬升角未知，一共9张图像
    initial_value = np.zeros(55)
    # initial_value[0: 4] = getInitialRotation(90, 0, 180)
    # initial_value[4: 7] = np.array([-0.1, 0., -0.2])
    my_args = (marker1, marker2, marker3, marker4, marker1GT, marker2GT, marker3GT, marker4GT, aligned_sonarFixatorPose)
    min_bounds, max_bounds = np.zeros(55), np.zeros(55)
    min_bounds[0: 4] = -1
    max_bounds[0: 4] = 1
    min_bounds[4: 7] = -1
    max_bounds[4: 7] = 1
    # min_bounds[7: 43] = -6.
    # max_bounds[7: 43] = 6.
    min_bounds[7: 55] = -6.
    max_bounds[7: 55] = 6.
    solution = least_squares(multiCaliEquations, initial_value, jac='3-point', args=my_args, ftol=3e-16, xtol=3e-16, gtol=3e-16)
    print(solution)
    q_w, q_x, q_y, q_z, p_x, p_y, p_z = solution.x[0: 7]
    Tf2s = np.eye(4, 4)
    # 填进旋转矩阵
    Tf2s[:3, :3] = np.array([
        [1 - 2 * (q_y ** 2 + q_z ** 2), 2 * (q_x * q_y - q_z * q_w), 2 * (q_x * q_z + q_y * q_w)],
        [2 * (q_x * q_y + q_z * q_w), 1 - 2 * (q_x ** 2 + q_z ** 2), 2 * (q_y * q_z - q_x * q_w)],
        [2 * (q_x * q_z - q_y * q_w), 2 * (q_y * q_z + q_x * q_w), 1 - 2 * (q_x ** 2 + q_y ** 2)]
    ])
    # 填进平移矩阵
    Tf2s[:3, -1] = np.array([p_x, p_y, p_z])
    print(Tf2s)