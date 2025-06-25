import os
import sys
import numpy as np
import torch
import torch.jit
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import math
import io
import base64
import tempfile #
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QHBoxLayout, QLabel, QFileDialog, QWidget, QProgressBar,
                             QStatusBar, QTabWidget, QSpinBox, QDoubleSpinBox,
                             QComboBox, QMessageBox, QTextEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QLineEdit, QGridLayout,
                             QGroupBox, QFrame, QInputDialog) # <-- Added QInputDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
import pandas as pd
import io # <-- Added for loading from memory
import base64 # <-- Added for key derivation



# --- Encryption/Decryption Dependencies ---
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("WARNING: 'cryptography' library not found. Decryption will fail.")


SALT_SIZE = 16
PBKDF2_ITERATIONS = 390000
# --- Name of the encrypted file to be embedded ---
EMBEDDED_ENCRYPTED_FILENAME = "encrypted_model.bin"
ORIGINAL_MODEL_TYPE = '.pth' # Or '.pth'

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

# --- Model Definition Function (Only needed if original type was .pth) ---
if ORIGINAL_MODEL_TYPE == '.pth':
    def get_instance_segmentation_model(num_classes):
        """
        创建用于实例分割的Mask R-CNN模型 (用于加载 .pth state_dict)
        ... (rest of the function remains the same as before) ...
        """
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
            return model
        except Exception as e:
            print(f"Error creating model from torchvision: {e}")
            try:
                import torchvision
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

                model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
                in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
                hidden_layer = 256
                model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
                return model
            except Exception as e2:
                print(f"Alternative import also failed: {e2}")
                QMessageBox.critical(None, "Model Creation Error",
                                     f"Failed to create model structure: {e}\nAlternative failed: {e2}\n"
                                     "Please check PyTorch/Torchvision installation and compatibility.")
                sys.exit(1)

# --- Resource Path Function (Keep As Is) ---
def resource_path(relative_path):
    """ 获取资源的绝对路径，适用于开发环境和PyInstaller打包环境 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Preprocessing, Detection, Visualization (Keep As Is) ---
def preprocess_image(image_path):
    # ... (same as before) ...
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    transform = Compose([ToTensor()])
    image_tensor = transform(original_image)
    return image_tensor, original_image

def detect_regions(model, image_tensor, device, threshold=0.5):
    # ... (same as before, including the isinstance check for torch.jit.ScriptModule) ...
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
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
    # ... (same as before) ...
     return math.sqrt(area / math.pi)

def visualize_results(image, prediction, threshold=0.5):
    # ... (same as before) ...
    vis_image = image.copy()
    height, width = image.shape[:2]
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    if prediction['masks'].numel() == 0:
        masks = np.empty((0, height, width), dtype=np.float32)
    else:
        masks = prediction['masks'].squeeze(1).cpu().numpy()

    if masks.ndim == 2 and len(boxes) == 1: # Handle single mask case
        masks = np.expand_dims(masks, axis=0)

    areas = []
    radii = []
    region_info = []

    if len(masks) > 0 and len(boxes) > 0:
        for i in range(len(boxes)):
            if i < len(masks):
                binary_mask = (masks[i] > 0.5).astype(np.uint8)
                area = np.sum(binary_mask)
                if area > 0:
                    radius = calculate_equivalent_radius(area)
                    region_info.append({
                        'index': i, 'area': area, 'radius': radius,
                        'box': boxes[i], 'mask': binary_mask, 'score': scores[i]
                    })

    region_info.sort(key=lambda x: x['area'], reverse=True)

    for region_num, region in enumerate(region_info, 1):
        area = region['area']
        radius = region['radius']
        binary_mask = region['mask']
        color = np.random.randint(50, 255, 3).tolist()
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

    count = len(region_info)
    total_area = sum(areas)
    avg_radius = sum(radii) / count if count > 0 else 0
    average_area = total_area / count if count > 0 else 0

    text_color = (255, 255, 255); bg_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX; scale = 0.8; thickness = 2; pad = 10
    texts = [f"Count: {count}"]
    y_offset = 30
    for text in texts:
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(vis_image, (pad, y_offset - h - pad//2), (pad + w + pad, y_offset + pad//2), bg_color, -1)
        cv2.putText(vis_image, text, (pad + pad//2, y_offset), font, scale, text_color, thickness)
        y_offset += h + pad + 5

    results = {'count': count, 'areas': areas, 'radii': radii, 'total_area': total_area,
               'average_area': average_area, 'average_radius': avg_radius,
               'region_info': region_info, 'visualization': vis_image}
    return results


# --- Processing Thread (Keep As Is) ---
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
            prediction = detect_regions(self.model, image_tensor, self.device, self.threshold)

            self.update_progress.emit(70)
            results = visualize_results(original_image, prediction, self.threshold)

            self.update_progress.emit(100)
            results['original_image'] = original_image
            self.processing_complete.emit(results)

        except FileNotFoundError as e:
             self.processing_error.emit(f"文件错误: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.processing_error.emit(f"处理时发生意外错误: {type(e).__name__} - {e}")


# --- Matplotlib Canvas (Keep As Is) ---
class MplCanvas(FigureCanvas):
    # ... (same as before) ...
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_alpha(0)
        self.axes = self.fig.add_subplot(111)
        self.axes.patch.set_alpha(0)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")

# --- Main Application Window (MODIFIED) ---
class RegionDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("区域检测器 (受保护)")
        self.setMinimumSize(1200, 800)
        self.model = None
        self.current_image_path = None
        self.current_results = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # ... (more robust device check) ...
        if torch.cuda.is_available():
            try:
                torch.cuda.get_device_name(0); self.device = torch.device('cuda')
            except RuntimeError:
                print("CUDA device found but not accessible, using CPU."); self.device = torch.device('cpu')
        else: self.device = torch.device('cpu')

        # --- MODIFIED: Load UI first, then prompt for password ---
        self.setup_ui() # Setup UI elements first
        self.statusBar().showMessage("等待密码...")

        # Prompt for password and load model AFTER UI is visible
        # Using QTimer to ensure the main event loop starts before showing the dialog
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._prompt_and_load_model) # Delay slightly


    # --- NEW: Prompt for password and trigger loading ---
    def _prompt_and_load_model(self):
        """Prompts user for password and attempts to load the encrypted model."""
        if not CRYPTO_AVAILABLE:
             QMessageBox.critical(self, "错误", "加密库 'cryptography' 未安装或无法加载。\n无法解密模型。")
             self.statusBar().showMessage("错误：缺少加密库")
             self._set_ui_state_post_load(success=False)
             return

        password, ok = QInputDialog.getText(self, "需要密码", "请输入模型密码:", QLineEdit.Password)

        if ok and password:
            self.statusBar().showMessage("正在解密和加载模型，请稍候...")
            QApplication.processEvents() # Update UI message
            success = self._load_encrypted_model(password)
            self._set_ui_state_post_load(success)
        elif ok and not password:
             QMessageBox.warning(self, "密码错误", "密码不能为空。")
             self.statusBar().showMessage("模型加载取消 (空密码)")
             self._set_ui_state_post_load(success=False)
        else:
            # User cancelled
            self.statusBar().showMessage("模型加载已取消。")
            self._set_ui_state_post_load(success=False)
            # Optionally, close the application if model is mandatory
            # self.close()
    def _load_encrypted_model(self, password):
        """Loads and decrypts the embedded model file using the provided password."""
        if not CRYPTO_AVAILABLE: return False

        temp_model_path = None  # Initialize variable to hold temp file path
        try:
            encrypted_model_path = resource_path(EMBEDDED_ENCRYPTED_FILENAME)
            print(f"Attempting to load encrypted model from: {encrypted_model_path}")

            if not os.path.exists(encrypted_model_path):
                raise FileNotFoundError(f"嵌入的加密模型文件未找到: {encrypted_model_path}")

            # 1. Read salt and encrypted data
            with open(encrypted_model_path, 'rb') as f_enc:
                salt = f_enc.read(SALT_SIZE)
                encrypted_data = f_enc.read()

            if len(salt) != SALT_SIZE:
                raise ValueError("Encrypted file is corrupted or too short (invalid salt).")

            # 2. Derive key
            print(f"Deriving key from password (using {PBKDF2_ITERATIONS} iterations)...")
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(), length=32, salt=salt,
                iterations=PBKDF2_ITERATIONS, backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            print("Key derived.")

            # 3. Decrypt data
            cipher_suite = Fernet(key)
            print("Attempting decryption...")
            decrypted_data = cipher_suite.decrypt(encrypted_data)
            print(f"Decryption successful. Data size: {len(decrypted_data)} bytes.")

            # 4. --- MODIFICATION START ---
            # Instead of using BytesIO, save decrypted data to a temporary file
            # Create a temporary file securely
            # 'delete=False' is important because we need to close it before torch loads it,
            # and then delete it manually afterwards.
            with tempfile.NamedTemporaryFile(delete=False, suffix=ORIGINAL_MODEL_TYPE) as temp_f:
                temp_model_path = temp_f.name  # Store the path
                temp_f.write(decrypted_data)
                print(f"Decrypted data written to temporary file: {temp_model_path}")

            # Now load the model from the temporary file path
            print(f"Loading model (Original Type: {ORIGINAL_MODEL_TYPE}) from temporary file...")

            if ORIGINAL_MODEL_TYPE == '.pt':
                self.model = torch.jit.load(temp_model_path, map_location=self.device)  # <-- Load from path
            elif ORIGINAL_MODEL_TYPE == '.pth':
                self.model = get_instance_segmentation_model(num_classes=2)  # Assuming num_classes=2
                # For .pth, loading from BytesIO *might* work, but consistency is good.
                # state_dict = torch.load(temp_model_path, map_location=self.device)
                # Or load from stream if preferred for .pth:
                with io.BytesIO(decrypted_data) as model_stream:
                    state_dict = torch.load(model_stream, map_location=self.device)

                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
                self.model = self.model.to(self.device)
                self.model.eval()
            else:
                raise ValueError(f"Unknown original model type specified: {ORIGINAL_MODEL_TYPE}")

            print("Model loaded successfully from temporary file.")
            del decrypted_data  # Free memory earlier if possible
            return True  # Success
            # --- MODIFICATION END ---

        except FileNotFoundError as e:
            # ... (error handling as before) ...
            self.model = None
            return False
        except InvalidToken:
            # ... (error handling as before) ...
            self.model = None
            return False
        except Exception as e:
            # ... (error handling as before) ...
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "加载错误", f"加载或解密内部模型时发生错误:\n{type(e).__name__}: {e}")
            self.model = None
            return False
        finally:
            # --- IMPORTANT: Clean up the temporary file ---
            if temp_model_path and os.path.exists(temp_model_path):
                try:
                    os.remove(temp_model_path)
                    print(f"Temporary model file deleted: {temp_model_path}")
                except Exception as e_del:
                    print(f"Warning: Could not delete temporary file {temp_model_path}: {e_del}")
            # Ensure key material is cleared if possible (though Python GC helps)
            key = None
            cipher_suite = None

    # --- NEW: Helper to set UI state after load attempt ---
    def _set_ui_state_post_load(self, success):
        """Updates status bar and enables/disables buttons based on model load success."""
        if success:
            self.statusBar().showMessage(f"模型已加载。使用设备: {self.device}")
            self.browse_image_btn.setEnabled(True)
            self.process_btn.setEnabled(False) # Only enable after image load
            self.save_btn.setEnabled(False)
        else:
            # Status message should already be set by the calling function
            self.browse_image_btn.setEnabled(False)
            self.process_btn.setEnabled(False)
            self.save_btn.setEnabled(False)


    # --- MODIFIED setup_ui: Removed model selection ---
    def setup_ui(self):
        """设置用户界面 (移除模型加载部分)"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Input Group (MODIFIED) ---
        input_group = QGroupBox("输入")
        input_layout = QGridLayout()
        input_group.setLayout(input_layout)

        # REMOVED: Model path widgets

        # Image Path (Row 0)
        input_layout.addWidget(QLabel("图像路径:"), 0, 0)
        self.image_path_edit = QLineEdit(); self.image_path_edit.setReadOnly(True)
        input_layout.addWidget(self.image_path_edit, 0, 1)
        self.browse_image_btn = QPushButton("浏览...")
        self.browse_image_btn.clicked.connect(self.load_image_dialog)
        self.browse_image_btn.setEnabled(False) # Initially disabled
        input_layout.addWidget(self.browse_image_btn, 0, 2)

        # Threshold (Row 1)
        input_layout.addWidget(QLabel("检测阈值:"), 1, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99); self.threshold_spin.setValue(0.5); self.threshold_spin.setSingleStep(0.05)
        input_layout.addWidget(self.threshold_spin, 1, 1)

        # Process Button (Row 1)
        self.process_btn = QPushButton("处理图像")
        self.process_btn.clicked.connect(self.process_current_image)
        self.process_btn.setEnabled(False) # Initially disabled
        input_layout.addWidget(self.process_btn, 1, 2)

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
        self.save_btn.setEnabled(False) # Initially disabled
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

        # --- Status Bar (Used by __init__ and loaders) ---


    # --- load_image_dialog (Keep As Is, checks self.model implicitly by button state) ---
    def load_image_dialog(self):
        # ... (same as before) ...
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "加载图像", "", "图像 (*.png *.jpg *.jpeg *.bmp *.tiff);;所有文件 (*)", options=options)
        if file_path:
            self.current_image_path = file_path
            self.image_path_edit.setText(file_path)
            self.statusBar().showMessage(f"图像已加载: {os.path.basename(file_path)}")
            try:
                img = cv2.imread(file_path)
                if img is None: raise IOError("无法使用OpenCV加载图像")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                self.original_canvas.axes.cla(); self.original_canvas.axes.imshow(img)
                self.original_canvas.axes.set_title("原始图像"); self.original_canvas.axes.axis('off')
                self.original_canvas.draw()

                self.result_canvas.axes.cla()
                self.result_canvas.axes.text(0.5, 0.5, '等待处理...', ha='center', va='center')
                self.result_canvas.axes.set_title(""); self.result_canvas.axes.axis('off')
                self.result_canvas.draw()

                self.results_table.setRowCount(0); self.summary_text.clear()
                # Enable processing ONLY IF model is also loaded (checked implicitly by initial state)
                self.process_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
            except Exception as e:
                 self.statusBar().showMessage(f"加载图像时出错: {e}")
                 QMessageBox.critical(self, "图像加载错误", f"加载图像失败: {e}")
                 self.process_btn.setEnabled(False); self.save_btn.setEnabled(False)

    # --- process_current_image (Keep As Is, but add model check) ---
    def process_current_image(self):
        # Explicit check for model state here
        if not self.model:
            QMessageBox.warning(self, "无法处理", "模型未能成功加载或解密，无法处理图像。")
            return
        if not self.current_image_path:
            QMessageBox.warning(self, "无法处理", "请先加载图像。")
            return

        # Disable buttons during processing
        # self.browse_model_btn.setEnabled(False) # Already removed
        self.browse_image_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        threshold = self.threshold_spin.value()

        self.processing_thread = ProcessingThread(self.model, self.current_image_path, threshold)
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.processing_error.connect(self.on_processing_error)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

        self.statusBar().showMessage("正在处理图像...")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    # --- on_processing_complete (Keep As Is) ---
    def on_processing_complete(self, results):
        # ... (same as before) ...
        self.current_results = results
        self.progress_bar.setValue(100)

        if 'original_image' in results:
            self.original_canvas.axes.cla(); self.original_canvas.axes.imshow(results['original_image'])
            self.original_canvas.axes.set_title("原始图像"); self.original_canvas.axes.axis('off')
            self.original_canvas.draw()

        self.result_canvas.axes.cla(); self.result_canvas.axes.imshow(results['visualization'])
        self.result_canvas.axes.set_title(f"检测到 {results['count']} 个区域"); self.result_canvas.axes.axis('off')
        self.result_canvas.draw()

        if 'region_info' in results:
            self.results_table.setRowCount(results['count'])
            for i, region in enumerate(results['region_info']):
                 self.results_table.setItem(i, 0, QTableWidgetItem(f"{i + 1}"))
                 self.results_table.setItem(i, 1, QTableWidgetItem(f"{region['area']:.2f}"))
                 self.results_table.setItem(i, 2, QTableWidgetItem(f"{region['radius']:.2f}"))
        else:
             self.results_table.setRowCount(results['count'])
             for i, (area, radius) in enumerate(zip(results['areas'], results['radii'])):
                self.results_table.setItem(i, 0, QTableWidgetItem(f"{i + 1}")); self.results_table.setItem(i, 1, QTableWidgetItem(f"{area:.2f}")); self.results_table.setItem(i, 2, QTableWidgetItem(f"{radius:.2f}"))

        summary = f"图像: {os.path.basename(self.current_image_path)}\n\n" \
                  f"检测到的区域数量: {results['count']}\n" \
                  f"总面积: {results['total_area']:.2f}\n" \
                  f"平均面积: {results['average_area']:.2f}\n" \
                  f"平均等效半径: {results['average_radius']:.2f}\n"
        self.summary_text.setText(summary)

        self.save_btn.setEnabled(True) # Enable save
        self.statusBar().showMessage("处理完成")

    # --- on_processing_error (Keep As Is) ---
    def on_processing_error(self, error_message):
        # ... (same as before) ...
        self.progress_bar.setValue(0)
        self.statusBar().showMessage(f"处理错误: {error_message}")
        QMessageBox.critical(self, "处理错误", error_message)
        self.save_btn.setEnabled(False)

    # --- on_processing_finished (MODIFIED - no model button) ---
    def on_processing_finished(self):
        """Called when the processing thread finishes."""
        # self.browse_model_btn.setEnabled(True) # Removed
        # Only enable image browse if model loaded successfully
        self.browse_image_btn.setEnabled(self.model is not None)
         # Only enable process if model is loaded and image is loaded
        self.process_btn.setEnabled(self.model is not None and self.current_image_path is not None)
        # Save button state handled by on_processing_complete/on_processing_error

    # --- save_results (Keep As Is) ---
    def save_results(self):
        # ... (same as before) ...
        if not self.current_results:
            QMessageBox.warning(self, "无法保存", "没有可保存的结果。请先处理图像。")
            return

        options = QFileDialog.Options()
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存结果的目录", "", options=options)
        if not save_dir: return

        try:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            img_path = os.path.join(save_dir, f"{base_name}_检测结果.png")
            self.result_canvas.fig.savefig(img_path, dpi=150, bbox_inches='tight')

            csv_path = os.path.join(save_dir, f"{base_name}_区域数据.csv")
            data = []
            if 'region_info' in self.current_results:
                 for i, region in enumerate(self.current_results['region_info']):
                     data.append({'区域编号': i + 1, '面积': region['area'], '等效半径': region['radius'], '置信度': region['score']})
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            summary_path = os.path.join(save_dir, f"{base_name}_摘要.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(self.summary_text.toPlainText())

            self.statusBar().showMessage(f"结果已成功保存到 {save_dir}")
            QMessageBox.information(self, "保存完成", f"可视化图像、CSV数据和摘要已保存到:\n{save_dir}")
        except Exception as e:
            self.statusBar().showMessage(f"保存结果时出错: {e}")
            QMessageBox.critical(self, "保存错误", f"保存结果失败: {e}")

# --- Main Execution Block (Keep As Is) ---
if __name__ == "__main__":
    # Check for crypto library early if possible
    if not CRYPTO_AVAILABLE:
         app_temp = QApplication(sys.argv) # Need an app instance for MessageBox
         QMessageBox.critical(None, "依赖错误",
                             "必需的加密库 'cryptography' 未找到。\n请安装: pip install cryptography")
         sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = RegionDetectorApp()
    window.show()
    sys.exit(app.exec_())
