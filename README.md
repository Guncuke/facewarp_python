# face_warp
一个简单的换脸程序

环境:py37 + dlib + opencv + scipy

下载dlib人脸形状检测器模型数据：shape_predictor_68_face_landmarks.dat.bz2，并解压在models文件夹下

通过dlib识别人脸68个特征点

通过delaunay三角剖分得到三角面片

通过仿射变换把每个三角面片仿射变换到目标图像上

采用逆向变换的方式

源图像和目标图像背景融合采用泊松融合

![target_screenshot_07 06 2022](https://user-images.githubusercontent.com/78009909/172281458-6a5565b2-bd8f-4b72-b8c8-67dbc03cd984.png)
