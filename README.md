## 圆形特征测量

### circle_measure.py
#### 测量步骤1.0
1. area_cal()进行参数的标定，确定裁剪尺寸，标准圆形参数，开始帧结束帧
2. crop_video()裁剪视频，如有特征遗失调整裁剪尺寸
3. read_vidio()执行测量功能，查看结果曲线，如有个别点突变，通过erro_frame()判断错误帧初步原因，通过erro_frame()限制

- 对于"test-v1.MOV"，平均像素半径：16.0 || 实测半径：7mm
#### 测量步骤1.1
1. basic_info()获得视频的基本信息
2. 去掉H.ROI，用外部软件裁剪视频到适合尺寸
3. 并用crop_video_fast()去除裁剪的黑色背景
4. 用area_cal()得到初步的参数（标定半径，标定圆心，标定周长）
5. 第一次运行read_vidio()，查看有问题的帧并解决
6. 运行area_cal()得到平均半径，结合实测半径得到单位像素实际尺寸
7. 运行read_vidio()得到数据与曲线
- **"test3000-120.mp4"**
  - SIZE = (480, 216, 0, 0) # test-v1.MOV # ROI尺寸 
  - CAL_RADIUS = 25.3 # 标定半径
  - CENTER = (105.6, 241.9) # 标定圆心
  - PERIMETER = 162.8 # 标定周长?
  - ALL_FRAME = 5189
  - L_PER_PIXEL = 7/(2 * 27.7) # 每像素的实际尺寸 