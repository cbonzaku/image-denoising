""" ce fichier contient la plupart des fonction implementer dans
les autres modules avec une interface graphic """

import csv
import os
import pickle
import random
import sys
import pandas as pd
import scipy
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QLabel, QVBoxLayout, QGraphicsDropShadowEffect, \
    QGraphicsOpacityEffect
from PyQt5.uic import loadUi
import matplotlib.pyplot as plt
import matplotlib
from skimage.restoration import denoise_wavelet
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import numpy as np
from pywt import waverec2, wavedec2, threshold
from scipy.stats import kurtosis
from scipy import optimize
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise

directory=os.getcwd() #lien pour manipulion dans l'aplication
image=None #l'image original
image2=2 #limage bruite
img_d=None #l'image debruite
filename = 'finale_model.sav' #nom du model

#chargement du modele
loaded_model = pickle.load(open(filename, 'rb'))

#lien de la base
path='dataBase'

#traitement du fichier CSV
f = open('finalmybe.csv','w')
writer = csv.writer(f)
header=['std','beta','noise','seuil']
processed=0
writer.writerow(header)



#'estimateur median
def estim_noise(HH):
    denom = scipy.stats.norm.ppf(0.75)
    sigma = np.median(np.abs(HH)) / denom
    return sigma

#utilisation du model pour pridir le seuil
def guess_threshohld(co,sigma):
    std=np.std(co)
    co2=co.flatten()
    beta = kurtosis(co2)
    d = {'std': std, 'beta': beta, 'noise': sigma}

    ser = pd.Series(data=d, index=['std', 'beta', 'noise'])

    seuil = loaded_model.predict(ser.values.reshape(1,-1))
    co1 = threshold(co, value=seuil)
    return co1

#debruitage avec la transforme d'ondelette en niveau 3
def den(r):
    original_extent = tuple(slice(s) for s in r.shape)
    coeffs = wavedec2(r, 'Haar', level=3)
    LL1, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH, HL, HH) = coeffs
    sigma=estim_noise(HH)
    LH3_ = guess_threshohld(LH3,sigma)
    HL3_= guess_threshohld(HL3,sigma)
    HH3_= guess_threshohld(HH3,sigma)
    LH2_= guess_threshohld(LH2,sigma)
    HL2_= guess_threshohld(HL2,sigma)
    HH2_= guess_threshohld(HH2,sigma)
    LH_= guess_threshohld(LH,sigma)
    HL_= guess_threshohld(HL,sigma)
    HH_= guess_threshohld(HH,sigma)

    r2=waverec2(( LL1,(LH3_,HL3_,HH3_),(LH2_,HL2_,HH2_),(LH_,HL_,HH_)), wavelet='Haar')[original_extent]

    return r2

#combini les canneau de l'image
def mergeRGB(r,g,b):
    rgb_uint8 = np.dstack((r, g, b))
    return rgb_uint8

#lire l'image
def getimg(imagead):
    global image
    image = cv2.imread(imagead)




#class qhi represente l'interface
class MainWindow(QDialog):
    #constricteur ou on met les button et les differentes composant de
    #interface comme les buttons
    def __init__(self):
        super(MainWindow,self).__init__()

        #charger le fichier de l'interface obtenue avec QT designer
        loadUi("untitled.ui",self)

        #ajouter les fonctions au button
        self.browse.clicked.connect(self.browsefiles)
        self.addNoise.clicked.connect(self.ajoutbruit)
        self.debruit.clicked.connect(self.d)

        #centrer le text
        self.Lb.setAlignment(Qt.AlignCenter)

        #difinir les effet visual
        shadow = QGraphicsDropShadowEffect()
        shadow1 = QGraphicsDropShadowEffect()
        shadow2 = QGraphicsDropShadowEffect()
        shadow3 = QGraphicsDropShadowEffect()
        shadow4 = QGraphicsDropShadowEffect()
        shadow5 = QGraphicsDropShadowEffect()
        # setting blur radius
        shadow.setBlurRadius(15)
        shadow1.setBlurRadius(15)
        shadow2.setBlurRadius(15)
        shadow3.setBlurRadius(15)
        shadow4.setBlurRadius(15)
        shadow5.setBlurRadius(15)
        self.groupBox.setGraphicsEffect(shadow5)
        self.groupBox_2.setGraphicsEffect(shadow1)
        self.groupBox_3.setGraphicsEffect(shadow2)
        self.groupBox_4.setGraphicsEffect(shadow3)
        self.groupBox_5.setGraphicsEffect(shadow4)
        self.bruite.setAlignment(Qt.AlignCenter)
        self.debruite.setAlignment(Qt.AlignCenter)
        self.pushButton.clicked.connect(self.draw)
        self.pushButton_2.clicked.connect(self.draw2)
        self.pushButton_3.clicked.connect(self.draw3)
        self.display.clicked.connect(self.affichage)



        self.generatDATA.clicked.connect(self.generate)
        self.opacity_effect = QGraphicsOpacityEffect()
        self.opacity_effect1 = QGraphicsOpacityEffect()
        self.opacity_effect2 = QGraphicsOpacityEffect()
        self.opacity_effect3 = QGraphicsOpacityEffect()
        self.opacity_effect4= QGraphicsOpacityEffect()
        self.opacity_effect5 = QGraphicsOpacityEffect()
        self.opacity_effect6 = QGraphicsOpacityEffect()
        self.opacity_effect7 = QGraphicsOpacityEffect()
        # setting opacity level
        self.opacity_effect.setOpacity(0.7)

        self.opacity_effect1.setOpacity(0.9)
        self.opacity_effect2.setOpacity(0.7)
        self.opacity_effect3.setOpacity(0.9)
        self.opacity_effect4.setOpacity(0.9)
        self.opacity_effect5.setOpacity(0.7)
        self.opacity_effect6.setOpacity(0.7)
        self.opacity_effect7.setOpacity(0.7)
        # adding opacity effect to the label
        self.groupBox_6.setGraphicsEffect(self.opacity_effect)
        self.groupBox_5.setGraphicsEffect(self.opacity_effect1)
        self.groupBox.setGraphicsEffect(self.opacity_effect2)
        self.groupBox_2.setGraphicsEffect(self.opacity_effect3)
        self.groupBox_3.setGraphicsEffect(self.opacity_effect4)
        self.groupBox_4.setGraphicsEffect(self.opacity_effect5)
        self.tableWidget.setGraphicsEffect(self.opacity_effect6)

        self.movie = QMovie("unnamed.gif")
        self.label_6.setMovie(self.movie)

        '''self.figure.tight_layout()
        self.gridLayout_2.addWidget(self.canvas)'''

        self.setAcceptDrops(True)


    #afficher les resultats clair dans une autre fentre
    def affichage(self):
        global image,image2,img_d



        if not img_d is None:
            i2 = cv2.imread(directory+'/noisy.jpg')
            i3 = cv2.imread(directory+'/debruit.jpg')
            f, axarr = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))

            r,g,b=cv2.split(image)
            i1=mergeRGB(b,g,r)
            r1, g1, b1 = cv2.split(image2)
            i2 = mergeRGB(b1, g1, r1)
            r2, g2, b2 = cv2.split(img_d)
            i3 = mergeRGB(b2, g2, r2)

            axarr[0].imshow(i1)

            axarr[0].set_title('image orginal\n')
            axarr[0].axis('off')
            axarr[1].imshow(i2)

            axarr[1].set_title('image bruite\n')
            axarr[1].axis('off')
            axarr[2].imshow(i3)

            axarr[2].axis('off')
            axarr[2].set_title('image debruite\n')
            plt.show()

    # boucle pour parcourir tout les image du dossier
    def generate(self):
        for i in reversed(range(self.gridLayout_2.count())):
            QtWidgets.QApplication.instance().processEvents()
            self.gridLayout_2.itemAt(i).widget().setParent(None)
        self.figure = Figure(figsize=(5, 3))
        # self.canvas = FigureCanvas(self.figure)
        # self.ax1 = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(plt.Figure(figsize=(15, 6)))
        self.ax = self.canvas.figure.subplots()
        self.ax.set(facecolor = "grey")
        self.ax.set_title('real time data generation')
        self.gridLayout_2.addWidget(self.canvas)
        global processed

        seuils = []

        ST = []
        ku = []
        l = 0
        for img in os.listdir(path):
            # lire l'image
            image = cv2.imread(os.path.join(path, img))
            image = img_as_float(image)
            sigma =  random.choice(np.arange(0.1,0.3,0.5))

            # ajouter du bruit
            image2 = random_noise(image, var=(sigma ** 2))
            r, g, b = cv2.split(image)
            r2, g2, b2 = cv2.split(image2)

            original_extent = tuple(slice(s) for s in r2.shape)  # utilise pour eviter la deformation d'image
            coeffs = wavedec2(g2, 'haar', level=3)  # performe la DWT
            LL1, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH, HL, HH) = coeffs
            # tablau des ecart-types
            stds = []
            # tablau des betas
            kur = []

            # boucle pour remplir les tablaux
            for i, a in enumerate(coeffs[1:]):
                for j in a:
                    stds.append(np.std(j))
                    ST.append(np.std(j))
                    k = kurtosis(j, axis=None)
                    ku.append(k)
                    kur.append(k)

            print(stds)
            print(kur)

            def treshold(param):
                QtWidgets.QApplication.instance().processEvents()
                LH3_ = threshold(LH3, value=abs(param[0]))
                HL3_ = threshold(HL3, value=abs(param[1]))
                HH3_ = threshold(HH3, value=abs(param[2]))
                LH2_ = threshold(LH2, value=abs(param[3]))
                HL2_ = threshold(HL2, value=abs(param[4]))
                HH2_ = threshold(HH2, value=abs(param[5]))
                LH_ = threshold(LH, value=abs(param[6]))
                HL_ = threshold(HL, value=abs(param[7]))
                HH_ = threshold(HH, value=abs(param[8]))
                # DWT inverse
                s = waverec2((LL1, (LH3_, HL3_, HH3_),
                              (LH2_, HL2_, HH2_),
                              (LH_, HL_, HH_)), wavelet='haar')[original_extent]
                psnr = peak_signal_noise_ratio(r, s)
                return psnr

            QtWidgets.QApplication.instance().processEvents()
            # initialization des seuils pour chaque composant de la DWT
            ini = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
            # trouver la millieur combinaison des seuil pour un maximum PSNR
            result1 = optimize.minimize(lambda x: -treshold(x), ini, method='powell'
                                        , options={'return_all': 'true', 'adaptive': 'true'}
                                        , bounds=(
                (0.0000000000001, 1), (0.0000000000001, 1), (0.0000000000001, 1), (0.0000000000001, 1),
                (0.0000000000001, 1), (0.0000000000001, 1), (0.0000000000001, 1), (0.0000000000001, 1),
                (0.0000000000001, 1)))
            QtWidgets.QApplication.instance().processEvents()

            row = []

            row.extend(result1.x)
            seuils.extend(result1.x)
            nimages=3

            # ecrire dans le fichier CSV
            self.tableWidget.setRowCount(len(row)*nimages)
            for i in range(0, 9):

                QtWidgets.QApplication.instance().processEvents()
                self.tableWidget.setItem(l, 0, QtWidgets.QTableWidgetItem(str(stds[i])))
                self.tableWidget.setItem(l, 1, QtWidgets.QTableWidgetItem(str(kur[i])))
                self.tableWidget.setItem(l, 2, QtWidgets.QTableWidgetItem(str(sigma)))
                self.tableWidget.setItem(l, 3, QtWidgets.QTableWidgetItem(str(row[i])))
                l = l+ 1
                self.update_plot(ST,seuils)
                writer.writerow([stds[i], kur[i], sigma, row[i]])
            processed += 1
            if processed >= 3:
                break


    # debruitage on click
    def d(self):
        global image, image2,img_d,directory
        #os.chdir(directory)
        filename = 'debruit.jpg'

        #bayeshrink
        if (self.radioButton_2.isChecked()):
            r, g, b = cv2.split(image2)
            r = denoise_wavelet(r, wavelet_levels=3
                                , method='BayesShrink', mode='soft', rescale_sigma=True)
            g = denoise_wavelet(g, wavelet_levels=3
                                , method='BayesShrink', mode='soft', rescale_sigma=True)
            b = denoise_wavelet(b, wavelet_levels=3, method='BayesShrink', mode='soft', rescale_sigma=True)
            img_d = mergeRGB(r, g, b)
            cv2.imwrite(filename, 255 * img_d)
            psnr_d = peak_signal_noise_ratio(image, img_d)
            self.psnrResult.setNum(psnr_d)
            self.set_image3(directory+'/debruit.jpg')

            #visuShrink
        if (self.radioButton.isChecked()):
            r, g, b = cv2.split(image2)
            r = denoise_wavelet(r, wavelet_levels=3
                                            , method='VisuShrink', mode='soft', rescale_sigma=True)
            g = denoise_wavelet(g, wavelet_levels=3
                                , method='VisuShrink', mode='soft', rescale_sigma=True)
            b = denoise_wavelet(b, wavelet_levels=3
                                , method='VisuShrink', mode='soft', rescale_sigma=True)
            img_d = mergeRGB(r, g, b)
            cv2.imwrite(filename, 255 * img_d)
            psnr_d = peak_signal_noise_ratio(image, img_d)
            self.psnrResult.setNum(psnr_d)
            self.set_image3(directory+'/debruit.jpg')

            #methode proposer
        if (self.radioButton_3.isChecked()):
            r,g,b= cv2.split(image2)
            r=den(r)
            g=den(g)
            b=den(b)
            img_d=mergeRGB(r,g,b)
            cv2.imwrite(filename, 255 * img_d)
            psnr_d = peak_signal_noise_ratio(image, img_d)
            self.psnrResult.setNum(psnr_d)
            self.set_image3(directory+'/debruit.jpg')


    #afficher les composant de la DWT et les histograme en niveau 1
    def draw(self):

        #vider le layout
        for i in reversed(range(self.gridLayout.count())):
            QtWidgets.QApplication.instance().processEvents()
            self.gridLayout.itemAt(i).widget().setParent(None)

        QtWidgets.QApplication.instance().processEvents()
        if not image is None:
            r, g, b = cv2.split(image)
            #difinir la forme de l'affichage
            LL1, (LH3, HL3, HH3) = wavedec2(r, 'Haar', level=1)
            QtWidgets.QApplication.instance().processEvents()
            self.figure = Figure(figsize=(8, 5))
            self.canvas = FigureCanvas(self.figure)
            titles = ['Approximation', ' Horizontal detail',
                      'Vertical detail', 'Diagonal detail']

            #affichage des composante et les histograme
            for i, a in enumerate([LL1, LH3, HL3, HH3]):
                QtWidgets.QApplication.instance().processEvents()
                ax = self.figure.add_subplot(2, 4, i + 1)
                QtWidgets.QApplication.instance().processEvents()
                ax.imshow(a, cmap=plt.cm.gray)
                QtWidgets.QApplication.instance().processEvents()
                ax.set_title(titles[i], fontsize=15)
                QtWidgets.QApplication.instance().processEvents()
                ax.set_xticks([])
                ax.set_yticks([])
                QtWidgets.QApplication.instance().processEvents()
            for i, a in enumerate([LH3, HL3, HH3]):
                QtWidgets.QApplication.instance().processEvents()
                ax = self.figure.add_subplot(2, 4, i + 6)
                QtWidgets.QApplication.instance().processEvents()
                ax.hist(a,bins=30,density=True)
                ax.set_xlabel('coefficents')
                ax.set_ylabel('probabilite')
                QtWidgets.QApplication.instance().processEvents()

                ax.set_yticks([])
                QtWidgets.QApplication.instance().processEvents()
            self.figure.tight_layout()
            QtWidgets.QApplication.instance().processEvents()
            self.gridLayout.addWidget(self.canvas)

    # afficher les composant de la DWT et les histograme en niveau 1
    def draw3(self):
        if not image is None:
            for i in reversed(range(self.gridLayout.count())):
                self.gridLayout.itemAt(i).widget().setParent(None)
            QtWidgets.QApplication.instance().processEvents()
            r, g, b = cv2.split(image)
            LL1, (LH3, HL3, HH3), (LH2, HL2, HH2),(LH,HL,HH) = wavedec2(r, 'Haar', level=3)
            QtWidgets.QApplication.instance().processEvents()
            self.figure = Figure(figsize=(8, 5))
            self.canvas = FigureCanvas(self.figure)
            titles = ['Approximation', ' Horizontal detail',
                      'Vertical detail', 'Diagonal detail']


            for i, a in enumerate([LL1, LH3, HL3, HH3]):
                QtWidgets.QApplication.instance().processEvents()
                ax = self.figure.add_subplot(2, 4, i + 1)
                QtWidgets.QApplication.instance().processEvents()
                ax.imshow(a, cmap=plt.cm.gray)
                QtWidgets.QApplication.instance().processEvents()
                ax.set_title(titles[i], fontsize=15)
                QtWidgets.QApplication.instance().processEvents()
                ax.set_xticks([])
                ax.set_yticks([])
                QtWidgets.QApplication.instance().processEvents()


            for i, a in enumerate([LH3, HL3, HH3]):
                QtWidgets.QApplication.instance().processEvents()
                ax = self.figure.add_subplot(2, 4, i + 6)
                QtWidgets.QApplication.instance().processEvents()
                ax.hist(a, bins=30, density=True)
                ax.set_xlabel('coefficents')
                ax.set_ylabel('probabilite')
                QtWidgets.QApplication.instance().processEvents()
                ax.set_yticks([])
                QtWidgets.QApplication.instance().processEvents()
            self.figure.tight_layout()
            self.gridLayout.addWidget(self.canvas)

    # afficher les composant de la DWT et les histograme en niveau 1
    def draw2(self):
        if not image is None:
            for i in reversed(range(self.gridLayout.count())):
                self.gridLayout.itemAt(i).widget().setParent(None)
                QtWidgets.QApplication.instance().processEvents()
            r, g, b = cv2.split(image)
            LL1, (LH3, HL3, HH3), (LH2, HL2, HH2) = wavedec2(r, 'Haar', level=2)
            QtWidgets.QApplication.instance().processEvents()
            self.figure = Figure(figsize=(8, 5))
            self.canvas = FigureCanvas(self.figure)
            titles = ['Approximation', ' Horizontal detail',
                  'Vertical detail', 'Diagonal detail']

            for i, a in enumerate([LL1, LH2, HL2, HH2]):
                QtWidgets.QApplication.instance().processEvents()
                ax = self.figure.add_subplot(2, 4, i + 1)
                QtWidgets.QApplication.instance().processEvents()
                ax.imshow(a, cmap=plt.cm.gray)
                QtWidgets.QApplication.instance().processEvents()
                ax.set_title(titles[i], fontsize=15)
                QtWidgets.QApplication.instance().processEvents()
                ax.set_xticks([])
                ax.set_yticks([])
                QtWidgets.QApplication.instance().processEvents()

            for i, a in enumerate([LH2, HL2, HH2]):
                QtWidgets.QApplication.instance().processEvents()
                ax = self.figure.add_subplot(2, 4, i + 6)
                QtWidgets.QApplication.instance().processEvents()
                ax.hist(a, bins=30, density=True)
                ax.set_xlabel('coefficents')
                ax.set_ylabel('probabilite')
                QtWidgets.QApplication.instance().processEvents()
                ax.set_yticks([])
                QtWidgets.QApplication.instance().processEvents()
            self.figure.tight_layout()
            self.gridLayout.addWidget(self.canvas)

    #ajouter du bruit a l'image
    def ajoutbruit(self):
        global image,image2,directory
        if not image is None:
            directory= directory
            filename = 'noisy.jpg'
            value=self.horizontalSlider.value()*0.01
            a=self.horizontalSlider.value()
            self.lcdNumber.display(a)

            image = img_as_float(image)

            b, g, r = cv2.split(image)
            image = mergeRGB(b, g, r)
            image2 = random_noise(image, var=(value ** 2))
            os.chdir(directory)
            cv2.imwrite(filename, 255*image2)
            self.set_image2(directory+'/noisy.jpg')
            psnr_bruit = peak_signal_noise_ratio(image, image2)
            self.psnrNoisy.setNum(psnr_bruit)
    #afficher une image dans une label
    def setPixmap(self , image):
        super().setPixmap(image)


    #choisir une image avec l'explorateur du system
    def browsefiles(self):
        global directory
        fname=QFileDialog.getOpenFileName(self, 'Open file', 'D:\codefirst.io\PyQt5 tutorials\Browse Files', 'Images (*.png, *.xmp *.jpg)')
        self.filename.setText(fname[0])
        self.set_image(fname[0])
        getimg(fname[0])

    #gestion des evenement
    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)
            event.accept()
            self.filename.setText(file_path)
            getimg(file_path)
        else:
            event.ignore()


    #affichage des images
    def set_image(self, file_path):
        self.Lb.setPixmap(QPixmap(file_path))

    def set_image2(self, file_path):
        self.bruite.setPixmap(QPixmap(file_path))

    def set_image3(self, file_path):
        self.debruite.setPixmap(QPixmap(file_path))

app=QApplication(sys.argv)
mainwindow=MainWindow()
widget=QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(1381)
widget.setFixedHeight(842)
widget.show()
sys.exit(app.exec_())