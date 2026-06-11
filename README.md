## 圆形特征测量

### circle_measure.py
测量步骤
1. area_cal()进行参数的标定，确定裁剪尺寸，标准圆形参数，开始帧结束帧
2. crop_video()裁剪视频，如有特征遗失调整裁剪尺寸
3. read_vidio()执行测量功能，查看结果曲线，如有个别点突变，通过erro_frame()判断错误帧初步原因，通过erro_frame()限制

- 对于"test-v1.MOV"，平均像素半径：16.0 || 实测半径：7mm