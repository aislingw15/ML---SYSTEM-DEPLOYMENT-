# Import necessary libraries
import sys
import os
import numpy as np
import cv2  # For image processing
import pydicom  # For handling DICOM medical image files
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import PyQt6 modules for building the GUI
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QProgressBar,
    QListWidget, QMessageBox, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QPalette, QColor
from PyQt6.QtCore import Qt, QTimer

# Load the pre-trained pneumonia detection model
MODEL_PATH = "pneumonia_model.h5"
model = load_model(MODEL_PATH)

# Main application class
class PneumoniaDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.dark_mode = False  # Flag to keep track of theme mode
        self.initUI()  # Initialize UI

    def initUI(self):
        # Setup window title, size, and drag/drop feature
        self.setWindowTitle("Medical Image Analysis 👨‍⚕️")
        self.setGeometry(100, 100, 400, 800)
        self.setAcceptDrops(True)

        # Create the main instruction label
        self.label = QLabel("Upload or Drag & Drop Medical Images 👨‍⚕️", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Upload button
        self.upload_button = QPushButton("📁 Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)

        # Label to display uploaded image
        self.image_label = QLabel("Drop image here 🖼️", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(500, 500)
        self.image_label.setStyleSheet("border: 2px dashed gray;")  # Dashed border for drop area

        # Progress bar to simulate analysis
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)

        # Label to show result of analysis
        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Label to show confidence score
        self.confidence_label = QLabel("Confidence: --% 📊", self)
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Analyze button (enabled after image is uploaded)
        self.analyze_button = QPushButton("⚡ Analyze Image", self)
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.animate_progress)

        # Button to upload another image
        self.new_image_button = QPushButton("📁 Upload New Image", self)
        self.new_image_button.setEnabled(False)
        self.new_image_button.clicked.connect(self.upload_image)

        # Button to export analysis report (optional feature placeholder)
        self.export_button = QPushButton("📋 Export Report", self)
        self.export_button.setEnabled(False)

        # Button to toggle between light and dark modes
        self.theme_button = QPushButton("🌓 Toggle Light/Dark Mode") 
        self.theme_button.clicked.connect(self.toggle_theme)

        # List to show history of image analysis
        self.history_list = QListWidget()
        self.history_list.setFixedHeight(100)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.image_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.export_button)
        layout.addWidget(QLabel("🗂️ Analysis History"))
        layout.addWidget(self.history_list)
        layout.addWidget(self.new_image_button)
        layout.addWidget(self.theme_button)

        self.setLayout(layout)

        # Initialize some variables
        self.image_path = ""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.progress_value = 0
        self.analysis_result = ""
        self.confidence = 0

    def upload_image(self):
        # Open file dialog to choose an image file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "", "All Image Files (*.dcm *.jpg *.jpeg *.png)")
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.analyze_button.setEnabled(True)

    def display_image(self, file_path):
        # Load and display selected image (DICOM or standard image formats)
        try:
            if file_path.endswith(".dcm"):
                dicom_data = pydicom.dcmread(file_path)
                image = dicom_data.pixel_array
            else:
                image = cv2.imread(file_path)

            if image is None:
                self.result_label.setText("❗ Error: Could not load image.")
                return

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (500, 500))
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

        except Exception as e:
            self.result_label.setText(f"❗ Error: {str(e)}")

    def animate_progress(self):
        # Start progress animation before showing result
        self.progress_value = 0
        self.progress_bar.setValue(0)
        self.result_label.setText("Analyzing... 🔬")
        self.timer.start(50)  # Timer updates progress bar

    def update_progress(self):
        # Simulate loading progress
        if self.progress_value < 100:
            self.progress_value += 5
            self.progress_bar.setValue(self.progress_value)
        else:
            # Stop progress and run model when complete
            self.timer.stop()
            result, confidence = self.run_pneumonia_detection(self.image_path)
            self.analysis_result = "✅ Pneumonia Detected" if result else "❌ No Pneumonia Detected"
            self.confidence = confidence
            self.result_label.setText(f"Result: {self.analysis_result}")
            self.confidence_label.setText(f"Confidence: {self.confidence:.2f}% 📊")
            self.new_image_button.setEnabled(True)
            self.export_button.setEnabled(True)
            self.history_list.addItem(f"{self.analysis_result} — {self.confidence:.2f}% Confidence")
            QMessageBox.information(self, "Analysis Complete", "✅ Analysis completed successfully!")

    def run_pneumonia_detection(self, image_path):
        # Process the image and make prediction using the model
        try:
            if image_path.endswith(".dcm"):
                dicom_data = pydicom.dcmread(image_path)
                image = dicom_data.pixel_array
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (150, 150))  # Resize to model input size
            image = np.stack((image,) * 3, axis=-1)  # Convert to 3 channels
            image = image / 255.0  # Normalize image
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            prediction = model.predict(image)[0][0]  # Get prediction
            confidence = prediction * 100  # Convert to percentage

            return prediction > 0.5, confidence  # Return result and confidence
        except Exception as e:
            print("Error during model prediction:", e)
            return False, 0

    def dragEnterEvent(self, event: QDragEnterEvent):
        # Accept drag event if it contains file URLs
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        # Handle dropped file and process it
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.dcm', '.jpg', '.jpeg', '.png')):
                self.image_path = file_path
                self.display_image(file_path)
                self.analyze_button.setEnabled(True)
                break

    def toggle_theme(self):
        # Toggle between dark and light mode
        self.dark_mode = not self.dark_mode

        if self.dark_mode:
            # Apply dark mode styles
            self.setStyleSheet("""
                QWidget { background-color: #2E2E2E; color: #FFFFFF; font-family: Arial; }
                QLabel#TitleLabel { font-size: 20px; font-weight: bold; margin-bottom: 10px; }
                QPushButton { background-color: #616161; color: white; border-radius: 5px; padding: 8px 16px; }
                QPushButton:hover { background-color: #757575; }
                QPushButton:disabled { background-color: #555; color: #aaa; }
                QProgressBar { border: 1px solid #757575; border-radius: 5px; text-align: center; }
                QProgressBar::chunk { background-color: #42A5F5; border-radius: 5px; }
                QListWidget { background-color: #424242; border: 1px solid #757575; padding: 5px; }
                QLabel { color: #FFFFFF; }
            """)
        else:
            # Apply light mode styles
            self.setStyleSheet("""
                QWidget { background-color: #F5F5F5; font-family: Arial; color: #333; }
                QLabel#TitleLabel { font-size: 20px; font-weight: bold; margin-bottom: 10px; }
                QPushButton { background-color: #1976D2; color: white; border-radius: 5px; padding: 8px 16px; }
                QPushButton:hover { background-color: #1565C0; }
                QPushButton:disabled { background-color: #B0BEC5; color: #ECEFF1; }
                QProgressBar { border: 1px solid #B0BEC5; border-radius: 5px; text-align: center; }
                QProgressBar::chunk { background-color: #64B5F6; border-radius: 5px; }
                QListWidget { background-color: #FFFFFF; border: 1px solid #B0BEC5; padding: 5px; }
            """)

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PneumoniaDetectorApp()
    window.show()
    sys.exit(app.exec())
