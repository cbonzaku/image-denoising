"""Dans ce module on utilise le modele entraine pour
    predir un seuil pour les diffirent composant de la transforme d'ondelette et
    utilser le seuil pour  debruiter l'image"""

import pickle
import numpy as np
import cv2
import pandas as pda
import scipy
from pywt import wavedec2, threshold, waverec2
from scipy.stats import kurtosis
from skimage import img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import denoise_wavelet
from skimage.util import random_noise
import pandas as pd
import matplotlib.pyplot as plt

"""fusionne les canneaux RGB"""
def mergeRGB(r,g,b):
    rgb_uint8 = np.dstack((r, g, b))
    return rgb_uint8


features = pda.read_csv('C:/Users/hp/finalmybe.csv') #load the csv file

#separe les donnes genere en:valeur qu'on veut estimer et les autreparametre
x1=pda.DataFrame(features, columns= ['std','beta','noise'])
y1=pda.DataFrame(features, columns= ['seuil'])

#prendre les 4000 premier lignes
x=x1.head(4000)
y=y1.head(4000)

#charger le modele sauvgarde
filename = 'finale_model23.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#prendre un echantion pour teste le modele
test_x=x1.last
test_y=y1.tail(500)

#load an image
image = cv2.imread('grey/6.jpg')
print(image.ndim)
#convertir le type d'image a float
image = img_as_float(image)

#decompose l'image en 3 canneau
b1,g1,r1=cv2.split(image)
image=mergeRGB(r1,g1,b1)

#ajouter du bruit
sigma =0.1
image2 = random_noise(image,mode='gaussian', var=(sigma**2))

#load model

loaded_model = pickle.load(open(filename, 'rb'))

#l'estimateur median du bruit
def estim_noise(HH):
    denom = scipy.stats.norm.ppf(0.75)
    sigma = np.median(np.abs(HH)) / denom
    return sigma

#deduire le seuil pour un composante de DWT
def guess_threshohld(co,sigma):
    std=np.std(co)
    co2=co.flatten()
    beta = kurtosis(co2)
    d = {'std': std, 'beta': beta, 'noise': sigma}

    ser = pd.Series(data=d, index=['std', 'beta', 'noise'])

    #pridir le seuil
    seuil = loaded_model.predict(ser.values.reshape(1,-1))
    co1 = threshold(co, value=seuil)
    return co1
#debruitage avec seuil
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

#separe l'image BGR en 3 canneaux
r, g, b = cv2.split(image2)

#preforme le debruitage et reconstruire l'image
b2 = den(b)
g2 = den(g)
r2 = den(r)
img_d = mergeRGB(r2, g2, b2)

#implementation du bayesShirnk et visuShrink
r3 = denoise_wavelet(r, wavelet_levels=3
                    , method='BayesShrink', mode='soft', rescale_sigma=True)
g3 = denoise_wavelet(g, wavelet_levels=3
                    , method='BayesShrink', mode='soft', rescale_sigma=True)
b3 = denoise_wavelet(b, wavelet_levels=3, method='BayesShrink', mode='soft', rescale_sigma=True)
img_B = mergeRGB(r3, g3, b3)
r4 = denoise_wavelet(r, wavelet_levels=3
                    , method='VisuShrink', mode='soft', rescale_sigma=True)
g4 = denoise_wavelet(g, wavelet_levels=3
                    , method='VisuShrink', mode='soft', rescale_sigma=True)
b4 = denoise_wavelet(b, wavelet_levels=3
                    , method='VisuShrink', mode='soft', rescale_sigma=True)
img_V = mergeRGB(r4, g4, b4)

#calculer le PSNR
psnr_B = peak_signal_noise_ratio(image, img_B)
psnr_V = peak_signal_noise_ratio(image, img_V)


#affichage du psnr et les resultats
psnr_d = peak_signal_noise_ratio(image, img_d)
psnr_bruit = peak_signal_noise_ratio(image, image2)
print(psnr_d,psnr_B,psnr_V)

f, axarr = plt.subplots(nrows=1,ncols=3,figsize=(10, 6))
axarr[0].imshow(image)
axarr[0].set_title('image orginal')
axarr[0].axis('off')
axarr[1].imshow(image2)
axarr[1].set_title('image bruite\nPSNR='+str(psnr_bruit))
axarr[1].axis('off')
axarr[2].imshow(img_d)
axarr[2].axis('off')
axarr[2].set_title('image debruite\nPSNR='+str(psnr_d))
plt.show()