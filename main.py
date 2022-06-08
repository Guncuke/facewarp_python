import os
import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay,tsearch
# from matplotlib import pyplot as plt

# 展示图片
def cv_show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 获得人脸60个特征点
def get_face_landmarks(image, face_detector, shape_predictor):
    """
    获取人脸标志，68个特征点
    :param image: image
    :param face_detector: dlib.get_frontal_face_detector
    :param shape_predictor: dlib.shape_predictor
    :return: np.array([[],[]]), 68个特征点
    """
    dets = face_detector(image, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found.")
        return None
    shape = shape_predictor(image, dets[0])
    face_landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return face_landmarks


# 得到人脸掩膜
def get_face_mask(image_size, face_landmarks):
    """
    获取人脸掩模
    :param image_size: 图片大小
    :param face_landmarks: 68个特征点
    :return: image_mask, 掩模图片
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    points = np.concatenate([face_landmarks[0:16], face_landmarks[26:17:-1]])
    cv2.fillPoly(img=mask, pts=[points], color=(255,255,255))
    return mask

def get_mask_center_point(image_mask):
    """
    获取掩模的中心点坐标
    :param image_mask: 掩模图片
    :return: 掩模中心
    """
    image_mask_index = np.argwhere(image_mask > 0)
    min = np.min(image_mask_index, axis=0)
    max = np.max(image_mask_index, axis=0)
    center_point = ((max[1] + min[1]) // 2, (max[0] + min[0]) // 2)
    return center_point

def affine_transform(source, source_feature_points, target, target_feature_points):
    """
        根据人脸68个点进行三角剖分变换
        :param source:源图像
        :param source_feature_points: 源图像特征点
        :param target:目标图像
        :param target_feature_points: 目标图像特征点
        :return: 变换后的源图像
        """
    # 三角剖分
    source_tri = Delaunay(source_feature_points)
    target_tri = Delaunay(target_feature_points)

    '''
    # 展示三角剖分
    source_RBG = source[...,::-1]
    target_RBG = target[...,::-1]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(source_RBG)
    plt.triplot(source_feature_points[:,0],source_feature_points[:,1],source_tri.simplices.copy())
    plt.plot(source_feature_points[:,0],source_feature_points[:,1],'o')
    plt.subplot(1,2,2)
    plt.imshow(target_RBG)
    plt.triplot(target_feature_points[:,0],target_feature_points[:,1],target_tri.simplices.copy())
    plt.plot(target_feature_points[:,0],target_feature_points[:,1],'o')
    plt.show()
    '''

    '''
    plt.triplot(target_feature_points[:,0],target_feature_points[:,1],target_tri.simplices.copy())
    plt.plot(target_feature_points[:,0],target_feature_points[:,1],'o')
    plt.show()
    '''
    # 记录来自哪个三角面片   所有用到Cv2的地方行列都要反过来
    d = target.shape
    index = np.zeros((d[1], d[0])).astype(int)
    for i in range(d[1]):
        for j in range(d[0]):
            index[i, j] = tsearch(target_tri, [i, j])

    # 每个三角形由哪三个顶点组成矩阵
    target_tri_arr = target_tri.simplices

    # 存储变换结果
    ans = np.zeros(target.shape, dtype=np.uint8)

    # 用来存储每个三角面片的仿射变换矩阵 加快速度
    affine_arr = dict()

    for i in range(d[0]):
        for j in range(d[1]):
            # 找到这个点对应的三个顶点， 中间图像的顶点
            # 对应第几个三角形
            tri_num = index[j, i]
            if tri_num != -1:
                point = np.array([[i], [j], [1]])
                if tri_num in affine_arr:
                    # 取出仿射变换矩阵
                    M = affine_arr[tri_num]
                    # 找到在源图像中的像素坐标
                    index_x_s, index_y_s, unuse = M @ point
                    index_x_s = np.round(index_x_s).astype(int)
                    index_y_s = np.round(index_y_s).astype(int)
                    ans[i, j] = source[index_x_s, index_y_s]
                else:
                    # 三个顶点(特征点提取时的第几个点）
                    A, B, C = target_tri_arr[tri_num]
                    # 构造矩阵
                    Ay = target_feature_points[A, 0]
                    Ax = target_feature_points[A, 1]
                    By = target_feature_points[B, 0]
                    Bx = target_feature_points[B, 1]
                    Cy = target_feature_points[C, 0]
                    Cx = target_feature_points[C, 1]
                    # Arr = np.array([[Ax, Bx, Cx], [Ay, By, Cy], [1, 1, 1]])
                    # 得到重心坐标
                    # aph = np.linalg.pinv(Arr) @ point
                    # 通过重心坐标，转换到源图像和目标图像，得到两个图像的像素值
                    Ay_s = source_feature_points[A, 0]
                    Ax_s = source_feature_points[A, 1]
                    By_s = source_feature_points[B, 0]
                    Bx_s = source_feature_points[B, 1]
                    Cy_s = source_feature_points[C, 0]
                    Cx_s = source_feature_points[C, 1]
                    # Arr_s = np.array([[Ax_s, Bx_s, Cx_s], [Ay_s, By_s, Cy_s], [1, 1, 1]])
                    # in_s = Arr_s @ aph
                    pts1 = np.float32([[Ax, Ay], [Bx, By], [Cx, Cy]])
                    pts2 = np.float32([[Ax_s, Ay_s], [Bx_s, By_s], [Cx_s, Cy_s]])
                    M = cv2.getAffineTransform(pts1, pts2)
                    M = np.insert(M, 2, [1,1,1], axis=0)
                    # 把三角面对应的仿射变换矩阵加入字典
                    affine_arr[tri_num] = M
                    # 找到在源图像中的像素坐标
                    index_x_s ,index_y_s, unuse = M @ point
                    index_x_s = np.round(index_x_s).astype(int)
                    index_y_s = np.round(index_y_s).astype(int)
                    ans[i, j] = source[index_x_s, index_y_s]
            else:
                continue
    return ans

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))

    models_folder_path = os.path.join(here, 'models')  # 模型保存文件夹
    predictor_path = os.path.join(models_folder_path, 'shape_predictor_68_face_landmarks.dat')  # 模型路径

    detector = dlib.get_frontal_face_detector()  # dlib的正向人脸检测器
    predictor = dlib.shape_predictor(predictor_path)  # dlib的人脸形状检测器

    source = cv2.imread('faces/source1.jpg')
    target = cv2.imread('faces/target1.jpg')

    # 获得人脸68个特征点
    source_feature_points = get_face_landmarks(source, detector, predictor)
    target_feature_points = get_face_landmarks(target, detector, predictor)

    # 获得目标图像人脸的掩膜
    # 因为通过三角面片转换到目标图像上，两个掩膜是重合的
    # source_face_mask = get_face_mask(source.shape, source_feature_points)
    target_face_mask = get_face_mask(target.shape, target_feature_points)

    # 目标图像掩膜中心
    target_face_center_point = get_mask_center_point(target_face_mask)

    '''
    cv_show('source_mask', source_face_mask)
    cv_show('taget_mask', target_face_mask)
    '''

    # 计算源图像变换后的图片
    ans = affine_transform(source, source_feature_points, target, target_feature_points)
    # 泊松融合
    seamless_im = cv2.seamlessClone(ans, target, mask=target_face_mask, p=target_face_center_point, flags=cv2.NORMAL_CLONE)
    cv_show('target', seamless_im)
