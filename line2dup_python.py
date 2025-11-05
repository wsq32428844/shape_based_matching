import sys
import ctypes
import cv2
import numpy as np
import os

# 加载line2Dup库
# 注意：需要先编译line2Dup库为共享库
# 例如：g++ -shared -o libline2dup.so line2Dup.cpp -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# 定义结构体
class Feature(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("label", ctypes.c_int),
        ("theta", ctypes.c_float)
    ]

class Template(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("tl_x", ctypes.c_int),
        ("tl_y", ctypes.c_int),
        ("pyramid_level", ctypes.c_int),
        ("features", ctypes.POINTER(Feature)),
        ("num_features", ctypes.c_int)
    ]

class Match(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("similarity", ctypes.c_float),
        ("class_id", ctypes.c_char_p),
        ("template_id", ctypes.c_int)
    ]

try:
    # 根据操作系统选择共享库扩展名
    if os.name == "posix":
        if sys.platform == "darwin":
            # macOS
            lib_ext = ".dylib"
        else:
            # Linux
            lib_ext = ".so"
    elif os.name == "nt":
        # Windows
        lib_ext = ".dll"
    else:
        raise Exception(f"不支持的操作系统: {os.name}")
    
    # 加载共享库
    lib_path = os.path.join(os.path.dirname(__file__), f"libline2dup{lib_ext}")
    libline2dup = ctypes.CDLL(lib_path)
    
    # 定义函数接口
    libline2dup.Detector_new.restype = ctypes.c_void_p
    libline2dup.Detector_new.argtypes = []
    
    libline2dup.Detector_addTemplate.restype = ctypes.c_int
    libline2dup.Detector_addTemplate.argtypes = [
        ctypes.c_void_p,  # detector
        ctypes.POINTER(ctypes.c_ubyte),  # image_data
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_int,  # channels
        ctypes.c_char_p,  # class_id
        ctypes.POINTER(ctypes.c_ubyte),  # mask_data
        ctypes.c_int,  # mask_width
        ctypes.c_int,  # mask_height
        ctypes.c_int   # num_features
    ]
    
    libline2dup.Detector_match.restype = ctypes.c_int
    libline2dup.Detector_match.argtypes = [
        ctypes.c_void_p,  # detector
        ctypes.POINTER(ctypes.c_ubyte),  # image_data
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_int,  # channels
        ctypes.c_float,  # threshold
        ctypes.POINTER(Match),  # matches
        ctypes.c_int   # max_matches
    ]
    
    libline2dup.Detector_delete.restype = None
    libline2dup.Detector_delete.argtypes = [ctypes.c_void_p]
    
    print("成功加载line2Dup库")
    
except Exception as e:
    print(f"加载line2Dup库失败: {e}")
    libline2dup = None

def mat_to_uchar_ptr(mat):
    """将OpenCV的Mat转换为uchar指针"""
    return mat.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

class Line2DupDetector:
    def __init__(self):
        if libline2dup is None:
            raise Exception("line2Dup库未加载")
        
        self.detector = libline2dup.Detector_new()
    
    def add_template(self, image, class_id="template", num_features=0):
        """添加模板"""
        if self.detector is None:
            raise Exception("检测器未初始化")
        
        # 转换图像为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 调整图像尺寸为16的倍数
        height, width = gray.shape
        new_width = ((width + 15) // 16) * 16
        new_height = ((height + 15) // 16) * 16
        gray = cv2.resize(gray, (new_width, new_height))
        
        # 获取新的图像尺寸和通道数
        height, width = gray.shape
        channels = 1  # 灰度图
        
        # 转换为uchar指针
        image_ptr = mat_to_uchar_ptr(gray)
        
        # 添加模板
        template_id = libline2dup.Detector_addTemplate(
            self.detector, image_ptr, width, height, channels, class_id.encode(),
            None, 0, 0, num_features
        )
        return template_id
    
    def match(self, image, threshold=0.5, max_matches=100):
        """匹配模板"""
        if self.detector is None:
            raise Exception("检测器未初始化")
        
        # 转换图像为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 调整图像尺寸为16的倍数
        height, width = gray.shape
        new_width = ((width + 15) // 16) * 16
        new_height = ((height + 15) // 16) * 16
        gray = cv2.resize(gray, (new_width, new_height))
        
        # 获取新的图像尺寸和通道数
        height, width = gray.shape
        channels = 1  # 灰度图
        
        # 转换为uchar指针
        image_ptr = mat_to_uchar_ptr(gray)
        
        # 分配匹配结果缓冲区
        matches_array = (Match * max_matches)()
        
        # 匹配
        num_matches = libline2dup.Detector_match(
            self.detector, image_ptr, width, height, channels, threshold,
            matches_array, max_matches
        )
        
        # 转换为Python列表
        matches = []
        for i in range(num_matches):
            match = matches_array[i]
            matches.append({
                "x": match.x,
                "y": match.y,
                "similarity": match.similarity,
                "class_id": match.class_id.decode(),
                "template_id": match.template_id
            })
        
        return matches
    
    def __del__(self):
        if self.detector is not None:
            libline2dup.Detector_delete(self.detector)
