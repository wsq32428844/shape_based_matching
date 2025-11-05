import sys
import os
import cv2
import numpy as np

# 确保可以找到line2dup_python模块
sys.path.append(os.path.dirname(__file__))

from line2dup_python import Line2DupDetector

def main():
    print("测试line2Dup Python包装器")
    
    try:
        # 初始化检测器
        detector = Line2DupDetector()
        print("成功初始化检测器")
        
        # 加载测试图像
        test_image = cv2.imread("test/case1/templ.png")
        if test_image is None:
            print("无法加载测试图像")
            return
        
        print(f"测试图像尺寸: {test_image.shape}")
        
        # 添加模板
        template_id = detector.add_template(test_image)
        print(f"成功添加模板，ID: {template_id}")
        
        # 加载匹配图像
        match_image = cv2.imread("test/case1/test.png")
        if match_image is None:
            print("无法加载匹配图像")
            return
        
        print(f"匹配图像尺寸: {match_image.shape}")
        
        # 匹配模板
        matches = detector.match(match_image, threshold=0.5)
        print(f"匹配结果: {matches}")
        
        print("测试完成")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
