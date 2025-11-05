import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from line2dup_python import Line2DupDetector

class ShapeBasedMatchingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shape Based Matching GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # 创建模板创建窗口
        self.template_create_tab = QWidget()
        self.create_template_create_tab()
        self.tab_widget.addTab(self.template_create_tab, "模板创建")
        
        # 创建模板匹配窗口
        self.template_match_tab = QWidget()
        self.create_template_match_tab()
        self.tab_widget.addTab(self.template_match_tab, "模板匹配")
        
        # 模板创建相关变量
        self.template_image = None
        self.template_image_path = None
        self.selection_rect = None
        self.dragging = False
        self.resizing = False
        self.drag_point = None
        self.resize_point = None
        
        # 模板匹配相关变量
        self.match_image = None
        self.match_image_path = None
        self.templates = []
        self.matches = []
        
        # 初始化检测器
        try:
            self.detector = Line2DupDetector()
        except Exception as e:
            QMessageBox.warning(self, "警告", f"初始化检测器失败: {e}")
            self.detector = None
    
    def create_template_create_tab(self):
        layout = QVBoxLayout()
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 打开图片按钮
        self.open_image_btn = QPushButton("打开图片")
        self.open_image_btn.clicked.connect(self.open_template_image)
        button_layout.addWidget(self.open_image_btn)
        
        # 创建模板按钮
        self.create_template_btn = QPushButton("创建模板")
        self.create_template_btn.clicked.connect(self.create_template)
        button_layout.addWidget(self.create_template_btn)
        
        layout.addLayout(button_layout)
        
        # 参数布局
        param_layout = QHBoxLayout()
        
        # Scale参数
        scale_group = QGroupBox("Scale")
        scale_layout = QVBoxLayout()
        self.scale_min = QLineEdit("0.8")
        self.scale_max = QLineEdit("1.2")
        self.scale_step = QLineEdit("0.1")
        scale_layout.addWidget(QLabel("Min:"))
        scale_layout.addWidget(self.scale_min)
        scale_layout.addWidget(QLabel("Max:"))
        scale_layout.addWidget(self.scale_max)
        scale_layout.addWidget(QLabel("Step:"))
        scale_layout.addWidget(self.scale_step)
        scale_group.setLayout(scale_layout)
        param_layout.addWidget(scale_group)
        
        # Angle参数
        angle_group = QGroupBox("Angle")
        angle_layout = QVBoxLayout()
        self.angle_min = QLineEdit("-10")
        self.angle_max = QLineEdit("10")
        self.angle_step = QLineEdit("1")
        angle_layout.addWidget(QLabel("Min:"))
        angle_layout.addWidget(self.angle_min)
        angle_layout.addWidget(QLabel("Max:"))
        angle_layout.addWidget(self.angle_max)
        angle_layout.addWidget(QLabel("Step:"))
        angle_layout.addWidget(self.angle_step)
        angle_group.setLayout(angle_layout)
        param_layout.addWidget(angle_group)
        
        layout.addLayout(param_layout)
        
        # 图片显示区域
        self.template_image_label = QLabel()
        self.template_image_label.setAlignment(Qt.AlignCenter)
        self.template_image_label.setFixedSize(800, 600)
        self.template_image_label.setStyleSheet("border: 1px solid black;")
        self.template_image_label.mousePressEvent = self.mouse_press_event
        self.template_image_label.mouseMoveEvent = self.mouse_move_event
        self.template_image_label.mouseReleaseEvent = self.mouse_release_event
        layout.addWidget(self.template_image_label)
        
        self.template_create_tab.setLayout(layout)
    
    def create_template_match_tab(self):
        layout = QVBoxLayout()
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 打开图片按钮
        self.open_match_image_btn = QPushButton("打开图片")
        self.open_match_image_btn.clicked.connect(self.open_match_image)
        button_layout.addWidget(self.open_match_image_btn)
        
        # 匹配按钮
        self.match_btn = QPushButton("匹配")
        self.match_btn.clicked.connect(self.match_templates)
        button_layout.addWidget(self.match_btn)
        
        layout.addLayout(button_layout)
        
        # 图片显示区域
        self.match_image_label = QLabel()
        self.match_image_label.setAlignment(Qt.AlignCenter)
        self.match_image_label.setFixedSize(800, 600)
        self.match_image_label.setStyleSheet("border: 1px solid black;")
        layout.addWidget(self.match_image_label)
        
        self.template_match_tab.setLayout(layout)
    
    def open_template_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图片", ".", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.template_image_path = file_path
            self.template_image = cv2.imread(file_path)
            self.display_image(self.template_image, self.template_image_label)
            self.selection_rect = None
    
    def open_match_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图片", ".", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.match_image_path = file_path
            self.match_image = cv2.imread(file_path)
            self.display_image(self.match_image, self.match_image_label)
    
    def display_image(self, image, label):
        if image is None:
            return
        
        # 调整图片大小以适应显示区域
        h, w, _ = image.shape
        label_w = label.width()
        label_h = label.height()
        
        scale = min(label_w / w, label_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # 转换为QImage
        qimage = QImage(resized_image.data, new_w, new_h, resized_image.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimage)
        
        label.setPixmap(pixmap)
    
    def mouse_press_event(self, event):
        if self.template_image is None:
            return
        
        # 检查是否点击了选择框
        if self.selection_rect:
            # 检查是否点击了旋转控制点（右上角）
            rect = self.selection_rect
            resize_point = QPoint(rect.right(), rect.top())
            if event.pos().distanceToPoint(resize_point) < 10:
                self.resizing = True
                self.resize_point = event.pos()
                return
            # 检查是否点击了平移控制点（左上角）
            drag_point = QPoint(rect.left(), rect.top())
            if event.pos().distanceToPoint(drag_point) < 10:
                self.dragging = True
                self.drag_point = event.pos() - QPoint(rect.left(), rect.top())
                return
        
        # 开始新的选择
        self.selection_rect = QRect(event.pos(), QSize())
        self.dragging = True
        self.drag_point = QPoint(0, 0)
    
    def mouse_move_event(self, event):
        if self.template_image is None:
            return
        
        if self.dragging:
            # 平移选择框
            if self.selection_rect.size().isEmpty():
                # 正在绘制新的选择框
                self.selection_rect.setBottomRight(event.pos())
            else:
                # 正在平移现有的选择框
                new_top_left = event.pos() - self.drag_point
                self.selection_rect.moveTopLeft(new_top_left)
            self.update_selection()
        elif self.resizing:
            # 调整选择框大小
            new_bottom_right = event.pos()
            self.selection_rect.setBottomRight(new_bottom_right)
            self.update_selection()
    
    def mouse_release_event(self, event):
        self.dragging = False
        self.resizing = False
    
    def update_selection(self):
        if self.template_image is None or self.selection_rect is None:
            return
        
        # 确保选择框在图片区域内
        image_rect = QRect(0, 0, self.template_image_label.pixmap().width(), self.template_image_label.pixmap().height())
        self.selection_rect = self.selection_rect.intersected(image_rect)
        
        # 显示选择框
        pixmap = self.template_image_label.pixmap().copy()
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.red, 2))
        painter.drawRect(self.selection_rect)
        
        # 绘制平移控制点（左上角）
        drag_point = QPoint(self.selection_rect.left(), self.selection_rect.top())
        painter.setBrush(QBrush(Qt.blue))
        painter.drawEllipse(drag_point, 5, 5)
        
        # 绘制旋转控制点（右上角）
        resize_point = QPoint(self.selection_rect.right(), self.selection_rect.top())
        painter.setBrush(QBrush(Qt.green))
        painter.drawEllipse(resize_point, 5, 5)
        
        painter.end()
        self.template_image_label.setPixmap(pixmap)
    
    def create_template(self):
        if self.template_image is None or self.selection_rect is None:
            QMessageBox.warning(self, "警告", "请先选择模板区域")
            return
        
        if self.detector is None:
            QMessageBox.warning(self, "警告", "检测器未初始化")
            return
        
        # 获取参数
        try:
            scale_min = float(self.scale_min.text())
            scale_max = float(self.scale_max.text())
            scale_step = float(self.scale_step.text())
            angle_min = float(self.angle_min.text())
            angle_max = float(self.angle_max.text())
            angle_step = float(self.angle_step.text())
        except ValueError:
            QMessageBox.warning(self, "警告", "参数必须为数字")
            return
        
        # 检查参数有效性
        if scale_min > scale_max or scale_step <= 0:
            QMessageBox.warning(self, "警告", "Scale参数无效")
            return
        
        if angle_min > angle_max or angle_step <= 0:
            QMessageBox.warning(self, "警告", "Angle参数无效")
            return
        
        # 获取选择区域
        pixmap = self.template_image_label.pixmap()
        h, w, _ = self.template_image.shape
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()
        
        scale = w / pixmap_w
        
        rect = self.selection_rect
        x1 = int(rect.left() * scale)
        y1 = int(rect.top() * scale)
        x2 = int(rect.right() * scale)
        y2 = int(rect.bottom() * scale)
        
        template_roi = self.template_image[y1:y2, x1:x2]
        
        # 添加模板
        try:
            template_id = self.detector.add_template(template_roi)
            self.templates.append(template_id)
            
            # 保存模板
            cv2.imwrite("template.jpg", template_roi)
            
            # 显示创建结果
            QMessageBox.information(self, "信息", f"模板创建成功，ID: {template_id}")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"创建模板失败: {e}")
    
    def match_templates(self):
        if self.match_image is None or not self.templates:
            QMessageBox.warning(self, "警告", "请先打开图片并创建模板")
            return
        
        if self.detector is None:
            QMessageBox.warning(self, "警告", "检测器未初始化")
            return
        
        try:
            # 调用line2Dup库进行匹配
            self.matches = self.detector.match(self.match_image)
            
            # 显示匹配结果
            result_image = self.match_image.copy()
            
            # 绘制匹配结果
            for match in self.matches:
                x, y = match["x"], match["y"]
                # 假设模板大小为选择区域的大小
                w, h = self.selection_rect.width(), self.selection_rect.height()
                similarity = match["similarity"]
                
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_image, f"{similarity:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.display_image(result_image, self.match_image_label)
            
            QMessageBox.information(self, "信息", f"匹配完成，找到 {len(self.matches)} 个匹配结果")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"匹配失败: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ShapeBasedMatchingGUI()
    gui.show()
    sys.exit(app.exec_())
