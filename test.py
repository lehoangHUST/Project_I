import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import classification
import process_data
import numpy as np

# Introduction author and project of author.
author = "Author:  LE HUY HOANG"
class_author = 'Class:  Class 01 - ET01-K63'
contact_email = 'Email:  lehuyhoang30122000@gmail.com'
instructor = 'Instructor:  NGUYEN HUY HOANG'
path_file = ""
predict_color = ""


class Example(QWidget):

    def __init__(self):

        super().__init__()
        self.title = "Project I"
        self.top = 100
        self.left = 100
        self.width = 1000
        self.height = 1500
        self.button_add = QPushButton('Add', self)
        self.button_KNN = QPushButton('K nearest neighbors', self)
        self.text_Edit = QTextEdit(self)
        self.initUI()

    def initUI(self):
        area_image = QLabel()

        # Label Image Lab.
        self.label_Icon = QLabel(self)
        self.label_Icon.setPixmap(QPixmap("C:/Users/Administrator/Pictures/edabk.jpg"))

        # Label Introduction Name.
        self.label_Text = QLabel("{0}\n{1}\n{2}\n{3}".format(author, instructor, class_author, contact_email), self)
        self.label_Text.setStyleSheet("background-color: white; border: 2px solid red")
        self.label_Text.setFont(QFont('Times', 17))

        # Label add image.
        self.label_Image = QLabel(self)
        self.label_Image.setStyleSheet("background-color: white; border: 2px solid gray")

        # Create GridLayout.
        grid = QGridLayout()

        # Each rows in GridLayout() space 60.
        grid.setSpacing(60)

        # Append function in grid.

        grid.addWidget(self.label_Icon, 0, 0)
        grid.addWidget(self.label_Text, 0, 1, 1, 20)
        grid.addWidget(self.button_add, 1, 0, 5, 1)
        grid.addWidget(self.button_KNN, 3, 0, 5, 1)
        grid.addWidget(QLabel('Predict color', self), 5, 0, 5, 1)
        grid.addWidget(self.text_Edit, 7, 0, 10, 1)
        grid.addWidget(self.label_Image, 1, 1, 18, 20)
        # Click button Add.
        self.button_add.clicked.connect(self.get_image_file)

        # Click algorithms KNN.
        self.button_KNN.clicked.connect(self.KNN)

        # Click algorithms Kmeans.
        self.setLayout(grid)

        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('Project I')

    # Take image in file => Label Image.
    def get_image_file(self):
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.jpg *.png *.jpeg)")
        imagePath = image[0]
        pixmap = QPixmap(imagePath)
        pixmap = pixmap.scaled(self.label_Image.width(), self.label_Image.height())
        # Print display path image file
        global path_file
        path_file = imagePath
        self.label_Image.setPixmap(pixmap)

    # Open camera and predict color image input.
    def open_camera(self):
        pass

    def KNN(self):
        img = process_data.Image(path_file)
        list = [img.blue, img.green, img.red]
        K_nearest_neighbors = classification.Knearestneighbor(k_nearest=1, norm=2)
        global predict_color
        predict_color = K_nearest_neighbors.predict_real_data(list)
        self.text_Edit.setPlainText(predict_color)

    def Kmeans(self):
        pass


# Function main => Run program.
def main():
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
