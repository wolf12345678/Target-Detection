import os
import sys
import numpy as np
import torch
import torch.jit # <-- Added for TorchScript loading
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QHBoxLayout, QLabel, QFileDialog, QWidget, QProgressBar,
                             QStatusBar, QTabWidget, QSpinBox, QDoubleSpinBox,
                             QComboBox, QMessageBox, QTextEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QLineEdit, QGridLayout,
                             QGroupBox, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
import pandas as pd


# --- Helper Classes and Functions (ToTensor, Compose, etc. - Keep As Is) ---
class ToTensor:
    """自定义ToTensor变换，用于兼容性"""
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            if pic.ndim == 2: pic = pic[:, :, np.newaxis]
            img = pic.transpose((2, 0, 1))
            img = img.astype(np.float32) / 255.0
            return torch.from_numpy(img)
        else:
            raise TypeError(f'pic应为ndarray，而不是{type(pic)}')

class Compose:
    """自定义Compose类，用于变换链"""
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms: img = t(img)
        return img

# --- Model Definition Function (Keep As Is) ---
def get_instance_segmentation_model(num_classes):
    """
    创建用于实例分割的Mask R-CNN模型 (用于加载 .pth state_dict)
    ... (rest of the function remains the same) ...
    """
    try:
        # Try importing from torchvision
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        # Load Mask R-CNN model
        model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes) # Use weights=None for newer torchvision
        # Or for older versions: model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
        return model
    except Exception as e:
        print(f"Error creating model from torchvision: {e}")
        # --- Fallback/Alternative (Keep As Is) ---
        try:
            import torchvision
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT') # Use weights='DEFAULT'
            # Or for older versions: model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, hidden_layer, num_classes
            )
            return model
        except Exception as e2:
            print(f"Alternative import also failed: {e2}")
            # Add more robust error handling or inform the user
            QMessageBox.critical(None, "Model Creation Error",
                                 f"Failed to create model structure: {e}\nAlternative failed: {e2}\n"
                                 "Please check PyTorch/Torchvision installation and compatibility.")
            sys.exit(1) # Or handle more gracefully

# --- Original Model Loading Function (For .pth State Dicts) ---
def load_model(model_path, num_classes=2, device=None):
    """从检查点文件加载训练好的模型参数 (state_dict)"""
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 1. Initialize the model structure
    model = get_instance_segmentation_model(num_classes)

    # 2. Load the weights (state dictionary)
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Handle potential keys mismatch (e.g., if saved with DataParallel)
        if list(state_dict.keys())[0].startswith('module.'):
             state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error loading state dict from {model_path}: {e}")

    model = model.to(device)
    model.eval()

    return model

# --- NEW: TorchScript Model Loading Function (For .pt files) ---
def load_torchscript_model(model_path, device=None):
    """加载 TorchScript 模型 (.pt file)"""
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    try:
        # Load the TorchScript model
        model = torch.jit.load(model_path, map_location=device)
        # TorchScript model is loaded directly to the specified device
        # and should already be in eval mode if saved correctly.
        # model.eval() # Usually not needed, but doesn't hurt
    except Exception as e:
         raise RuntimeError(f"Error loading TorchScript model from {model_path}: {e}")

    return model

# --- Preprocessing, Detection, Visualization (Keep As Is) ---
def preprocess_image(image_path):
    # ... (rest of the function remains the same) ...
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    transform = Compose([ToTensor()])
    image_tensor = transform(original_image)
    return image_tensor, original_image

def detect_regions(model, image_tensor, device, threshold=0.5):
    # ... (rest of the function remains the same) ...
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        # Check if the model is a TorchScript model (ScriptModule)
        # TorchScript models are called directly
        # Standard PyTorch models might need the input in a list
        if isinstance(model, torch.jit.ScriptModule):
             prediction = model(image_tensor.unsqueeze(0))[0] # Add batch dim for TorchScript
        else:
             prediction = model([image_tensor])[0] # Standard PyTorch model expects a list

    keep = prediction['scores'] >= threshold
    filtered_prediction = {
        'boxes': prediction['boxes'][keep],
        'labels': prediction['labels'][keep],
        'scores': prediction['scores'][keep],
        'masks': prediction['masks'][keep]
    }
    return filtered_prediction


def calculate_equivalent_radius(area):
    # ... (rest of the function remains the same) ...
    return math.sqrt(area / math.pi)

def visualize_results(image, prediction, threshold=0.5):
    # ... (rest of the function remains the same) ...
    vis_image = image.copy()
    height, width = image.shape[:2]
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    # Ensure masks are handled correctly even if empty
    if prediction['masks'].numel() == 0:
        masks = np.empty((0, height, width), dtype=np.float32)
    else:
        masks = prediction['masks'].squeeze(1).cpu().numpy() # Remove channel dim if present

    if masks.ndim == 2 and len(boxes) == 1: # Handle single mask case
        masks = np.expand_dims(masks, axis=0)

    areas = []
    radii = []
    region_info = []

    # Check if there are any masks to process
    if len(masks) > 0 and len(boxes) > 0:
        for i in range(len(boxes)):
            # Ensure mask index is valid
            if i < len(masks):
                binary_mask = (masks[i] > 0.5).astype(np.uint8) # Use 0.5 threshold for mask binarization
                area = np.sum(binary_mask)
                if area > 0: # Only process regions with non-zero area
                    radius = calculate_equivalent_radius(area)
                    region_info.append({
                        'index': i, 'area': area, 'radius': radius,
                        'box': boxes[i], 'mask': binary_mask, 'score': scores[i]
                    })

    # Sort by area
    region_info.sort(key=lambda x: x['area'], reverse=True)

    # Draw sorted regions
    for region_num, region in enumerate(region_info, 1):
        area = region['area']
        radius = region['radius']
        binary_mask = region['mask']
        color = np.random.randint(50, 255, 3).tolist() # Brighter colors
        colored_mask = np.zeros_like(vis_image)
        for c in range(3): colored_mask[:, :, c] = color[c]
        masked_area = (binary_mask[:, :, np.newaxis] * colored_mask).astype(np.uint8)
        vis_image = cv2.addWeighted(vis_image, 1, masked_area, 0.5, 0)
        x1, y1, x2, y2 = region['box'].astype(int)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_image, f"#{region_num}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        areas.append(area)
        radii.append(radius)

    # Calculate stats
    count = len(region_info) # Count only regions added to region_info
    total_area = sum(areas)
    avg_radius = sum(radii) / count if count > 0 else 0
    average_area = total_area / count if count > 0 else 0

    # Add text overlay
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    pad = 10
    texts = [
        f"数量: {count}",
        f"总面积: {total_area:.1f}",
        f"平均半径: {avg_radius:.2f}"
    ]
    y_offset = 30
    for text in texts:
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(vis_image, (pad, y_offset - h - pad//2), (pad + w + pad, y_offset + pad//2), bg_color, -1)
        cv2.putText(vis_image, text, (pad + pad//2, y_offset), font, scale, text_color, thickness)
        y_offset += h + pad + 5 # Add spacing

    results = {
        'count': count, 'areas': areas, 'radii': radii,
        'total_area': total_area, 'average_area': average_area,
        'average_radius': avg_radius, 'region_info': region_info, # Use sorted info
        'visualization': vis_image
    }
    return results


# --- Processing Thread (Keep As Is, but check detect_regions call) ---
class ProcessingThread(QThread):
    # ... (signals remain the same) ...
    update_progress = pyqtSignal(int)
    processing_complete = pyqtSignal(dict)
    processing_error = pyqtSignal(str)

    def __init__(self, model, image_path, threshold=0.5):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.threshold = threshold
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def run(self):
        try:
            self.update_progress.emit(10)
            image_tensor, original_image = preprocess_image(self.image_path)

            self.update_progress.emit(40)
            # Pass model directly to detect_regions, it handles the type check now
            prediction = detect_regions(self.model, image_tensor, self.device, self.threshold)

            self.update_progress.emit(70)
            results = visualize_results(original_image, prediction, self.threshold)

            self.update_progress.emit(100)
            results['original_image'] = original_image # Add original image to results
            self.processing_complete.emit(results)

        except FileNotFoundError as e:
             self.processing_error.emit(f"文件错误: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc() # Print full traceback to console for debugging
            self.processing_error.emit(f"处理时发生意外错误: {type(e).__name__} - {e}")


# --- Matplotlib Canvas (Keep As Is) ---
class MplCanvas(FigureCanvas):
    # ... (class remains the same) ...
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_alpha(0) # Transparent background for fig
        # self.fig.tight_layout() # Can sometimes cause issues, adjust padding manually if needed
        self.axes = self.fig.add_subplot(111)
        self.axes.patch.set_alpha(0) # Transparent background for axes
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;") # Ensure canvas widget is transparent

# --- Main Application Window ---
class RegionDetectorApp(QMainWindow):
    # ... (__init__ remains mostly the same) ...
    def __init__(self):
        super().__init__()
        self.setWindowTitle("区域检测器")
        self.setMinimumSize(1200, 800)
        self.model = None
        self.model_path = None
        self.current_image_path = None
        self.current_results = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Check CUDA availability more robustly
        if torch.cuda.is_available():
            try:
                torch.cuda.get_device_name(0) # Try accessing device
                self.device = torch.device('cuda')
            except RuntimeError:
                print("CUDA device found but not accessible, using CPU.")
                self.device = torch.device('cpu')
        else:
             self.device = torch.device('cpu')

        # Setup UI
        self.setup_ui()

        # Use the METHOD to get the status bar
        self.statusBar().showMessage(f"就绪。使用设备: {self.device}")

    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Input Group (Keep As Is) ---
        input_group = QGroupBox("输入")
        input_layout = QGridLayout()
        input_group.setLayout(input_layout)
        # Model Path
        input_layout.addWidget(QLabel("模型路径:"), 0, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        input_layout.addWidget(self.model_path_edit, 0, 1)
        self.browse_model_btn = QPushButton("浏览...")
        self.browse_model_btn.clicked.connect(self.load_model_dialog)
        input_layout.addWidget(self.browse_model_btn, 0, 2)
        # Image Path
        input_layout.addWidget(QLabel("图像路径:"), 1, 0)
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        input_layout.addWidget(self.image_path_edit, 1, 1)
        self.browse_image_btn = QPushButton("浏览...")
        self.browse_image_btn.clicked.connect(self.load_image_dialog)
        self.browse_image_btn.setEnabled(False)
        input_layout.addWidget(self.browse_image_btn, 1, 2)
        # Threshold
        input_layout.addWidget(QLabel("检测阈值:"), 2, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99); self.threshold_spin.setValue(0.5); self.threshold_spin.setSingleStep(0.05)
        input_layout.addWidget(self.threshold_spin, 2, 1)
        # Process Button
        self.process_btn = QPushButton("处理图像")
        self.process_btn.clicked.connect(self.process_current_image)
        self.process_btn.setEnabled(False)
        input_layout.addWidget(self.process_btn, 2, 2)
        main_layout.addWidget(input_group)

        # --- Output Group (Keep As Is) ---
        output_group = QGroupBox("输出")
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)
        self.tab_widget = QTabWidget()
        # Viz Tab
        self.viz_tab = QWidget()
        viz_layout = QVBoxLayout(self.viz_tab)
        images_layout = QHBoxLayout()
        # Original Image
        original_frame = QFrame(); original_frame.setFrameShape(QFrame.StyledPanel)
        original_layout = QVBoxLayout(original_frame)
        original_layout.addWidget(QLabel("原始图像", alignment=Qt.AlignCenter))
        self.original_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        original_layout.addWidget(self.original_canvas)
        # Result Image
        result_frame = QFrame(); result_frame.setFrameShape(QFrame.StyledPanel)
        result_layout = QVBoxLayout(result_frame)
        result_layout.addWidget(QLabel("检测结果", alignment=Qt.AlignCenter))
        self.result_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        result_layout.addWidget(self.result_canvas)
        images_layout.addWidget(original_frame)
        images_layout.addWidget(result_frame)
        viz_layout.addLayout(images_layout)
        # Save Button
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        viz_layout.addWidget(self.save_btn)
        # Results Tab
        self.results_tab = QWidget()
        results_layout = QVBoxLayout(self.results_tab)
        # Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["区域 #", "面积", "等效半径"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        # Summary View
        results_layout.addWidget(QLabel("结果摘要:"))
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(200)
        results_layout.addWidget(self.summary_text)
        # Add Tabs
        self.tab_widget.addTab(self.viz_tab, "可视化")
        self.tab_widget.addTab(self.results_tab, "详细结果")
        output_layout.addWidget(self.tab_widget)
        main_layout.addWidget(output_group)

        # --- Progress Bar (Keep As Is) ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # --- Status Bar (No manual creation needed, use inherited method) ---
        # self.statusBar = QStatusBar() # REMOVED - Use self.statusBar() method
        # self.setStatusBar(self.statusBar) # REMOVED - QMainWindow handles this

    # --- MODIFIED: load_model_dialog ---
    def load_model_dialog(self):
        """打开文件对话框加载模型 (支持 .pth 和 .pt)"""
        options = QFileDialog.Options()
        # MODIFIED: Update filter to include .pt
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载模型", "",
            "模型文件 (*.pt *.pth);;TorchScript (*.pt);;PyTorch State Dict (*.pth);;所有文件 (*)",
            options=options
        )

        if file_path:
            self.statusBar().showMessage(f"正在从 {os.path.basename(file_path)} 加载模型...")
            QApplication.processEvents() # Update UI

            try:
                # <<<--- NEW LOGIC: Check file extension --->>>
                if file_path.lower().endswith('.pt'):
                    print("检测到 TorchScript 文件 (.pt)，尝试加载...")
                    self.model = load_torchscript_model(file_path, device=self.device)
                    model_type = "TorchScript"
                elif file_path.lower().endswith('.pth'):
                    print("检测到 PyTorch State Dict 文件 (.pth)，尝试加载...")
                    # Assuming num_classes=2, adjust if needed or make it configurable
                    self.model = load_model(file_path, num_classes=2, device=self.device)
                    model_type = "State Dict"
                else:
                    # Optional: Handle unsupported file types more explicitly
                    raise ValueError("不支持的模型文件类型。请选择 .pt 或 .pth 文件。")
                # <<<------------------------------------->>>

                self.model_path = file_path
                self.model_path_edit.setText(file_path)
                # Use the METHOD statusBar()
                self.statusBar().showMessage(f"{model_type} 模型成功加载自 {os.path.basename(file_path)}")
                self.browse_image_btn.setEnabled(True)
                self.process_btn.setEnabled(False) # Disable process until image is loaded
                self.save_btn.setEnabled(False) # Disable save until processed

            except Exception as e:
                 # Use the METHOD statusBar()
                self.statusBar().showMessage(f"加载模型时出错: {str(e)}")
                QMessageBox.critical(self, "模型加载错误", f"加载模型失败: {str(e)}")
                self.model = None # Ensure model is None on failure
                self.browse_image_btn.setEnabled(False)
                self.process_btn.setEnabled(False)
                self.save_btn.setEnabled(False)

    # --- load_image_dialog (Keep As Is, but maybe clear canvases better) ---
    def load_image_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载图像", "", "图像 (*.png *.jpg *.jpeg *.bmp *.tiff);;所有文件 (*)",
            options=options
        )
        if file_path:
            self.current_image_path = file_path
            self.image_path_edit.setText(file_path)
            self.statusBar().showMessage(f"图像已加载: {os.path.basename(file_path)}")
            try:
                img = cv2.imread(file_path)
                if img is None:
                     raise IOError("无法使用OpenCV加载图像")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Clear and display original image
                self.original_canvas.axes.cla() # Clear axes
                self.original_canvas.axes.imshow(img)
                self.original_canvas.axes.set_title("原始图像")
                self.original_canvas.axes.axis('off')
                self.original_canvas.draw()

                # Clear result canvas
                self.result_canvas.axes.cla() # Clear axes
                self.result_canvas.axes.text(0.5, 0.5, '等待处理...', ha='center', va='center') # Placeholder text
                self.result_canvas.axes.set_title("")
                self.result_canvas.axes.axis('off')
                self.result_canvas.draw()

                # Clear table and summary
                self.results_table.setRowCount(0)
                self.summary_text.clear()

                # Enable processing
                self.process_btn.setEnabled(True)
                self.save_btn.setEnabled(False) # Disable save until processed

            except Exception as e:
                 self.statusBar().showMessage(f"加载图像时出错: {e}")
                 QMessageBox.critical(self, "图像加载错误", f"加载图像失败: {e}")
                 self.process_btn.setEnabled(False)
                 self.save_btn.setEnabled(False)


    # --- process_current_image (Keep As Is) ---
    def process_current_image(self):
        if not self.model or not self.current_image_path:
            QMessageBox.warning(self, "无法处理", "请先加载模型和图像。")
            return

        self.browse_model_btn.setEnabled(False)
        self.browse_image_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.save_btn.setEnabled(False) # Disable save during processing
        self.progress_bar.setValue(0) # Reset progress

        threshold = self.threshold_spin.value()

        self.processing_thread = ProcessingThread(self.model, self.current_image_path, threshold)
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.processing_error.connect(self.on_processing_error)
        self.processing_thread.finished.connect(self.on_processing_finished) # Re-enable buttons on finish regardless of success/error
        self.processing_thread.start()

        self.statusBar().showMessage("正在处理图像...")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    # --- on_processing_complete (Keep As Is) ---
    def on_processing_complete(self, results):
        self.current_results = results
        self.progress_bar.setValue(100)

        # Display original (might be redundant if load_image does it, but safe)
        if 'original_image' in results:
            self.original_canvas.axes.cla()
            self.original_canvas.axes.imshow(results['original_image'])
            self.original_canvas.axes.set_title("原始图像")
            self.original_canvas.axes.axis('off')
            self.original_canvas.draw()

        # Display result visualization
        self.result_canvas.axes.cla()
        self.result_canvas.axes.imshow(results['visualization'])
        self.result_canvas.axes.set_title(f"检测到 {results['count']} 个区域")
        self.result_canvas.axes.axis('off')
        self.result_canvas.draw()

        # Update results table (use sorted region_info if available)
        if 'region_info' in results:
            self.results_table.setRowCount(results['count'])
            for i, region in enumerate(results['region_info']):
                 # Use region data which is sorted
                self.results_table.setItem(i, 0, QTableWidgetItem(f"{i + 1}")) # Use sorted index
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{region['area']:.2f}"))
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{region['radius']:.2f}"))
        else: # Fallback if region_info not present (should be)
             self.results_table.setRowCount(results['count'])
             for i, (area, radius) in enumerate(zip(results['areas'], results['radii'])):
                self.results_table.setItem(i, 0, QTableWidgetItem(f"{i + 1}"))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{area:.2f}"))
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{radius:.2f}"))


        # Update summary text
        summary = f"图像: {os.path.basename(self.current_image_path)}\n\n"
        summary += f"检测到的区域数量: {results['count']}\n"
        summary += f"总面积: {results['total_area']:.2f}\n"
        summary += f"平均面积: {results['average_area']:.2f}\n"
        summary += f"平均等效半径: {results['average_radius']:.2f}\n"
        self.summary_text.setText(summary)

        self.save_btn.setEnabled(True) # Enable save after successful processing
        self.statusBar().showMessage("处理完成")

    # --- on_processing_error (Keep As Is) ---
    def on_processing_error(self, error_message):
        self.progress_bar.setValue(0) # Reset progress on error
        self.statusBar().showMessage(f"处理错误: {error_message}")
        QMessageBox.critical(self, "处理错误", error_message)
        self.save_btn.setEnabled(False) # Ensure save is disabled on error

    # --- NEW: Slot to re-enable buttons when thread finishes ---
    def on_processing_finished(self):
        """Called when the processing thread finishes (success or error)."""
        self.browse_model_btn.setEnabled(True)
        # Only enable image browse if model loaded successfully
        self.browse_image_btn.setEnabled(self.model is not None)
         # Only enable process if model and image are loaded
        self.process_btn.setEnabled(self.model is not None and self.current_image_path is not None)
        # Save button state handled by on_processing_complete/on_processing_error

    # --- save_results (Keep As Is, maybe improve filename) ---
    def save_results(self):
        if not self.current_results:
            QMessageBox.warning(self, "无法保存", "没有可保存的结果。请先处理图像。")
            return

        options = QFileDialog.Options()
        # Suggest a default directory or remember last used? For now, just open dialog.
        save_dir = QFileDialog.getExistingDirectory(
            self, "选择保存结果的目录", "", options=options
        )

        if not save_dir:
            return

        try:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            # Save visualization
            img_path = os.path.join(save_dir, f"{base_name}_检测结果.png")
            # Use matplotlib to save the figure shown in the canvas
            # Need to be careful if canvas DPI differs from desired save DPI
            self.result_canvas.fig.savefig(img_path, dpi=150, bbox_inches='tight') # Save with decent resolution

            # Save CSV data (using sorted region_info)
            csv_path = os.path.join(save_dir, f"{base_name}_区域数据.csv")
            data = []
            if 'region_info' in self.current_results:
                 for i, region in enumerate(self.current_results['region_info']):
                     data.append({
                         '区域编号': i + 1, # Use sorted index
                         '面积': region['area'],
                         '等效半径': region['radius'],
                         '置信度': region['score'] # Add score if available
                     })
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility

            # Save summary text
            summary_path = os.path.join(save_dir, f"{base_name}_摘要.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(self.summary_text.toPlainText()) # Write the content of the text edit

            self.statusBar().showMessage(f"结果已成功保存到 {save_dir}")
            QMessageBox.information(self, "保存完成", f"可视化图像、CSV数据和摘要已保存到:\n{save_dir}")

        except Exception as e:
            self.statusBar().showMessage(f"保存结果时出错: {e}")
            QMessageBox.critical(self, "保存错误", f"保存结果失败: {e}")


# --- Main Execution Block (Keep As Is) ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = RegionDetectorApp()
    window.show()
    sys.exit(app.exec_())

