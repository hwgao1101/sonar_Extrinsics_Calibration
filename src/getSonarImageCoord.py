import cv2
import numpy as np
import pickle

'''
保存7张声纳图像中marker点在图像上的坐标
'''

def drawMarkerOnImage(rawImageDir, marker1ImageCoord, marker2ImageCoord, marker3ImageCoord, marker4ImageCoord):
    for i in range(marker1ImageCoord.shape[0]):
        img = cv2.imread(f'{rawImageDir}/{i + 1}/img1.jpg', cv2.IMREAD_COLOR)
        print(img.shape)
        cv2.circle(img, (marker1ImageCoord[i, 0], marker1ImageCoord[i, 1]), 5, (0, 255, 0), 1)
        cv2.circle(img, (marker2ImageCoord[i, 0], marker2ImageCoord[i, 1]), 5, (0, 255, 0), 1)
        cv2.circle(img, (marker3ImageCoord[i, 0], marker3ImageCoord[i, 1]), 5, (0, 255, 0), 1)
        cv2.circle(img, (marker4ImageCoord[i, 0], marker4ImageCoord[i, 1]), 5, (0, 255, 0), 1)
        cv2.imwrite(f'{rawImageDir}/{i + 1}/img1WithMarker.jpg', img)
        print("has save down")

def getRangeAndTheta(marker1ImageCoord, marker2ImageCoord, marker3ImageCoord, marker4ImageCoord):
    height = 707
    width = 512
    # 把图像（x, y）的坐标点转成（r, \theta）
    # 使用的高频，水平偏转角60度， 最大探测距离2m
    max_Range = 2  # 单位m
    horizontal_Angle = 60  # 这里的度数是角度制
    r_resolution = max_Range / height
    theta_resolution = horizontal_Angle / width

    # marker1
    Range1 = marker1ImageCoord[:, 1] * r_resolution
    theta1 = (marker1ImageCoord[:, 0] - width / 2) * theta_resolution
    marker1 = np.column_stack((Range1, 90 - theta1))

    # marker2
    Range2 = marker2ImageCoord[:, 1] * r_resolution
    theta2 = (marker2ImageCoord[:, 0] - width / 2) * theta_resolution
    marker2 = np.column_stack((Range2, 90 - theta2))

    # marker3
    Range3 = marker3ImageCoord[:, 1] * r_resolution
    theta3 = (marker3ImageCoord[:, 0] - width / 2) * theta_resolution
    marker3 = np.column_stack((Range3, 90 - theta3))

    # marke4
    Range4 = marker4ImageCoord[:, 1] * r_resolution
    theta4 = (marker4ImageCoord[:, 0] - width / 2) * theta_resolution
    marker4 = np.column_stack((Range4, 90 - theta4))

    return marker1, marker2, marker3, marker4

if __name__ == '__main__':

    # 手动获取像素点的坐标, 第7张和第8张声纳图像有问题，所以这里给删除掉
    marker1ImageCoord = np.array([[179, 226], [171, 228], [175, 227], [176, 226], [177, 225], [171, 227], [202, 219], [194, 209], [256, 206], [184, 209], [232, 203], [214, 203]]) # [246, 470], [263, 458],
    marker2ImageCoord = np.array([[216, 224], [211, 226], [214, 225], [215, 224], [218, 223], [211, 224], [244, 218], [235, 206], [302, 205], [229, 207], [278, 202], [261, 202]]) # [266, 467], [284, 455],
    marker3ImageCoord = np.array([[162, 251], [158, 253], [160, 253], [160, 251], [162, 251], [156, 252], [189, 245], [182, 233], [243, 232], [177, 236], [220, 229], [205, 228]]) # [244, 503], [260, 489],
    marker4ImageCoord = np.array([[243, 261], [239, 263], [240, 262], [242, 261], [244, 261], [238, 262], [272, 256], [269, 243], [327, 245], [264, 245], [307, 241], [294, 239]]) # [285, 508], [304, 493],

    rawImageDir = '/home/titanium/ubuntu/sonarCalibration/final/final_biaoding/first_bag/rawImage'
    # 把每张声纳图像提取出来的marker点绘制在图像上
    drawMarkerOnImage(rawImageDir, marker1ImageCoord, marker2ImageCoord, marker3ImageCoord, marker4ImageCoord)
    marker1, marker2, marker3, marker4 = getRangeAndTheta(marker1ImageCoord, marker2ImageCoord, marker3ImageCoord, marker4ImageCoord)

    # 将声纳图像上marker点的数据传过去
    markerCoordData = {'marker1': marker1, 'marker2': marker2, 'marker3': marker3, 'marker4': marker4}
    with open('./markerCoordData.pkl', 'wb') as file:
        pickle.dump(markerCoordData, file)
