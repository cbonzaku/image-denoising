"""ce module est utilise pour genere les donnes a partir de la base
 de donnes des images ,puis sauvgarde les donnes dans un fichier CSV"""

import os
import random
import cv2
import csv
import scipy
from matplotlib import pyplot
from pywt import waverec2, threshold, wavedec2
from scipy import optimize
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
from scipy.stats import kurtosis
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

#preparer un fichier csv pour stoker les donnes
path='dataBase'  #lien du fichier des images
f = open('test1.csv','w') #creation du fichier
writer = csv.writer(f) #prepare un writer pour ecrir dans le fichier
header=['std','beta','noise','seuil'] #les noms des parametre genere
processed=0 #compteur des image pour arrete la generation
writer.writerow(header)

#une fonction de debruitage qui utilise le seuil comme paramtre et elle retourn la valeur du psnr
def treshold(param):
    global r, LL1, LH3, HL3, HH3, LH2, HL2, HH2, LH, HL, HH
    LH3_ = threshold(LH3, value=abs(param[0]))
    HL3_ = threshold(HL3, value=abs(param[1]))
    HH3_ = threshold(HH3, value=abs(param[2]))
    LH2_ = threshold(LH2, value=abs(param[3]))
    HL2_ = threshold(HL2, value=abs(param[4]))
    HH2_ = threshold(HH2, value=abs(param[5]))
    LH_ = threshold(LH, value=abs(param[6]))
    HL_ = threshold(HL, value=abs(param[7]))
    HH_ = threshold(HH, value=abs(param[8]))

    #DWT inverse
    s = waverec2((LL1, (LH3_, HL3_, HH3_),
                  (LH2_, HL2_, HH2_),
                  (LH_, HL_, HH_)), wavelet='haar')[original_extent]
    psnr = peak_signal_noise_ratio(r, s)
    return psnr

#la valeur du bruit
sigma =0.1

#une boucle pour parcourir tout les images dans le dossier
for sigma in np.arange(0.1,0.35,0.05):
    print(sigma)
    #boucle pour parcourir tout les image du dossier
    for img in os.listdir(path):
        #lire l'image
        image = cv2.imread(os.path.join(path,img))
        image= img_as_float(image)


        #ajouter du bruit
        image2 = random_noise(image,mode='gaussian', var=(sigma**2))
        r,g,b =cv2.split(image)
        r2,g2,b2 =cv2.split(image2)

        # utilise pour eviter la deformation d'image
        original_extent = tuple(slice(s) for s in r2.shape)

        # performe la DWT
        coeffs = wavedec2(g2, 'haar', level=3)
        LL1,(LH3,HL3,HH3),(LH2,HL2,HH2),(LH,HL,HH) = coeffs

        #tablau des ecart-types
        stds=[]
        #tablau des betas
        kur=[]

    #boucle pour remplir les tablaux
        for i,a in enumerate(coeffs[1:]):
            for j in a :
                stds.append(np.std(j))

                k = kurtosis(j,axis=None)
                kur.append(k)


        print(stds)
        print(kur)

        #initialization des seuils pour chaque composant de la DWT
        ini=[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
        #trouver la millieur combinaison des seuil pour un maximum PSNR
        result1 = optimize.minimize(lambda x: -treshold(x), ini, method='powell'
                                , options={'return_all': 'true', 'adaptive': 'true'}
                                ,bounds=((0.00000001,1),(0.00000001,1),(0.000000001,1),(0.000000001,1),(0.000000001,1),(0.000000001,1),(0.000000001,1),(0.000000001,1),(0.000000001,1)))


        row=[]

        row.extend(result1.x)
        print(row)

        #ecrire dans le fichier CSV
        for i in range(0,9):
            writer.writerow([stds[i],kur[i],sigma,row[i]])
        processed+=1
        if processed >= 200:
            break


