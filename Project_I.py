from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import classification
import process_data
import sys


class MainWindow(QMainWindow):
    # Create init
    def __init__(self):
        super().__init__()
        self.title = "Project I"
        self.top = 100
        self.left = 100
        self.width = 1000
        self.height = 1500
        self.button_add = QPushButton('Add', self)
        self.button_camera = QPushButton('Camera', self)
        self.button_algorithms = QPushButton('Algorithms', self)
        self.InitWindow()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create image Laboratory.
        self.label = QLabel(self)
        self.label.setPixmap(QPixmap("C:/Users/Administrator/Pictures/edabk.jpg"))
        self.label.setGeometry(0, 0, 255, 200)

        # Button_add_Picture
        self.button_add.setGeometry(0, 200, 250, 50)
        self.button_add.clicked.connect(self.get_image_file)

        self.labelImage = QLabel(self)
        self.labelImage.setGeometry(250, 200, 505, 400)

        # Button_camera
        self.button_camera.setGeometry(0, 350, 250, 50)
        self.button_camera.clicked.connect(self.open_camera)

        # Button_algorithms
        self.button_algorithms.setGeometry(0, 500, 250, 50)
        self.button_algorithms.clicked.connect(self.algorithms)

    def get_image_file(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.jpg)")
        imagePath = image[0]
        pixmap = QPixmap(imagePath)
        self.labelImage.setPixmap(pixmap)

    def open_camera(self):
        pass

    def algorithms(self):
        pass

    def paintEvent(self, event):
        """
            # Draw line to two parts :
            # Part 1: Introduction author code and instructor
            # Part 2:
            # + Show image need predict and time to run algorithms
            # + Function open camera, choose algorithms, add picture in computer.

        """
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 1, Qt.SolidLine))
        painter.drawLine(0, 200, self.width, 200)
        painter.setPen(QPen(Qt.red, 1, Qt.SolidLine))
        painter.drawLine(250, 200, 250, self.height)


def main():
    App = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(App.exec())


if __name__ == "__main__":
    main()
