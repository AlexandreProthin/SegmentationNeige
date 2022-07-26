# ----------------------------------------------------------------------------------------
# ------------------------     Fonctions pour la segmentation    -------------------------
#
#   Organisation :
# 1. ensemble des fonctions de segmentation
# 2. fonctions d'évaluation des performances
# 3. Autre
#
# ----------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import cv2 as cv
import os 
import random
import glob
import sys
import tensorflow as tf
import natsort
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from keras.models import load_model
from sklearn.cluster import KMeans
from PIL import Image, ImageOps
from socket import SO_VM_SOCKETS_BUFFER_SIZE
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from random import shuffle
from keras_unet.models import custom_unet
from skimage.io import imread, imshow
from sklearn.metrics import f1_score
from matplotlib.path import Path


# ----------------------------------------------------------------------------------------
# --------------------  1. ensemble des fonctions de segmentation  -----------------------
# 
#            entrée : l'image à segmenter avec eventuellement des paramètre 
#            sortie : masque de segmentation Vrai = neige / Faux = sans neige
#
# ----------------------------------------------------------------------------------------

# fonction de seuillage simple 
def segmentTreshold(image, seuil):
    """ 
    image: image à seuiller
    seuil: seuil des treshold

    return: masque résultant de la segmentation
    """
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1,th1 = cv.threshold(gray, seuil, 255, cv.THRESH_BINARY)
    return th1.astype(bool)

# fonction de seuillage automatique 
def segmentOTSU(img):
    """ 
    image: image à seuiller

    return: masque résultant de la segmentation
    """
    # Otsu's thresholding
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret2,th2 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return th2.astype(bool)

# fonction de seuillage RGB
def segmentRGB(image, seuilNeige):
    """ 
    image: image à seuiller
    seuilNeige: Niveau d'intensité à partir duquel on considère que c'est de la neige

    return kPixelsNeige, image: retourne le nombre de pixels classifiés en tant que neige et l'image segmentée
    """
    seuilGris = 60
    img = np.zeros((1008, 1920))
    # segmentation par seuil sur les lignes
    for i in range(len(image)):
        # sur les colones
        for j in range(len(image[0])):
            # sur les 3 couleurs
            # calcul des distances RG et RB et BG
            r = image[i][j][0]
            g = image[i][j][1]
            b = image[i][j][2]
            #seuilGris: seuil interne à la fonction à partir duquel on considère qu'un triplet RGB représente un niveau de gris
            estUnNiveauDeGris = abs(r-b) > seuilGris or abs(g-b) > seuilGris or abs(g-r) > seuilGris
            if not(estUnNiveauDeGris): # si vrai c'est un niveau de gris
                if np.mean(image[i][j]) > seuilNeige: # si vrai c'est de la neige, ie: si la valeur moyenne des RGB est suppérieur au seil
                    img[i][j] = 255
    return img.astype(bool)

# segmentation dans l'espace 3d hsv (mieux que rgb)
def segmentSpace(image):
    """
    entrée: image à segmenter 
    sortie: masque de segmentation
    """
    hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    light_white = (0, 0, 100)
    # dark_white = (145, 60, 255)
    dark_white = (170, 90, 255)
    final_mask = cv.inRange(hsv_image, light_white, dark_white)
    return final_mask.astype(bool)

# fonction k-Means
def segmentkMeans(img, n_clusters, dir):
    """
    entrée: image à segmenter, nombre de classe à trouver, direction de la racine
    sortie: masque de segmentation
    """
    # mise en forme des données 
    df_img = image_to_pandas(img)
    df_img.head(5)

    #application du K-means
    kmeans = KMeans(n_clusters, random_state = 10).fit(df_img)
    result = kmeans.labels_.reshape(img.shape[0],img.shape[1])

    # récupération du modèle de reconnaissance
    labels = []
    with open(dir +"/Modèles_trained/Classif_clusters/labels.txt", "r") as f:
        for line in f:
            labels.append(line.strip()[2:])
    f.close()
    model = load_model(dir +"/Modèles_trained/Classif_clusters/keras_model.h5")
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # attribution des clusters à la bonne classe
    """ découpe, prédiction, rangement """
    masques = []
    for i in range(n_clusters):
        tempImg = img.copy()
        tempImg[:, :, 0] = tempImg[:, :, 0]*(result==[i])
        tempImg[:, :, 1] = tempImg[:, :, 1]*(result==[i])
        tempImg[:, :, 2] = tempImg[:, :, 2]*(result==[i])
        # width, height = len(tempImg[0]), len(tempImg)
        PIL_image = Image.fromarray(np.uint8(tempImg)).convert('RGB')
        size = (224, 224)
        resized = ImageOps.fit(PIL_image, size, Image.ANTIALIAS)
        image_array = np.asarray(resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        prediction = prediction.tolist()[0]
        prediction = normalize(prediction)

        if max(prediction) > 80 and labels[np.argmax(prediction)] == "clusterNeige":
            masques.append(result==[i])

    if len(masques) != 0:
        # regrouppement des clusters appartenant à la même classes
        segmentationMask = masques[0]
        for i in range(len(masques)):
            segmentationMask = masques[i] | segmentationMask
    else: 
        segmentationMask = np.zeros((1008, 1920)).astype(bool)
    return segmentationMask

# fonction DBSCAN
def segmentedDBSCAN(img, eps, min_samples):
    """
    entrée: image à segmenter, eps, min_samples
    sortie: masque de segmentation
    """
    labimg = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    n = 0
    while(n<4):
        labimg = cv.pyrDown(labimg)
        n = n+1

    feature_image=np.reshape(labimg, [-1, 3])
    rows, cols, chs = labimg.shape

    db = DBSCAN(eps, min_samples, metric = 'euclidean',algorithm ='auto')
    db.fit(feature_image)
    labels = db.labels_
    return np.reshape(labels, [rows, cols])

# fonction segmentation par patchs via SVM
def segmentedSVM(im, dir, patchSize = 9):
    """
    entrée: image à segmenter, patchSize = taille du patch à classifier 
            dir = root directory
            tailles de patch dispo : 25, 9, 5
    sortie: masque de segmentation
    """

    def getTrainningRef(patchSize):
        """ patchSize = 5, 9 ou 25 """  
        trainImagettesNeige = [] 
        trainImagettesSansNeige = [] 
        os.chdir(f'{dir}/data/Supervise/{patchSize}/')

        # sélection des imagettes positives
        imgs_path = os.listdir('Avec_neige')
        for i in imgs_path:
            trainImagettesNeige.append(cv.imread(f'Avec_neige/{i}'))
        random.Random(1337).shuffle(trainImagettesNeige)

        # sélection des imagettes négatives
        imgs_path = os.listdir('Sans_neige')
        for i in imgs_path:
            trainImagettesSansNeige.append(cv.imread(f"Sans_neige/{i}"))
        random.Random(1337).shuffle(trainImagettesNeige)

        return trainImagettesSansNeige, trainImagettesNeige

    def estDeLaNeige(img):
        """
        entrée: image à classifier
        sortie: booléen vrai = neige / faux = sansNeige
        """
        return clf.predict([cv.cvtColor(img, cv.COLOR_BGR2GRAY).flatten()]) == 1

    trainImagettesSansNeige, trainImagettesNeige = getTrainningRef(patchSize)
    X, y = [], []
    for img in trainImagettesSansNeige[::]:
        X.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY).flatten())
        y.append(0) # label sans neige

    for img in trainImagettesNeige[::]:
        X.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY).flatten())
        y.append(1)# label avec neige

    print("nb img for trainning =",len(y)/2)
    # split : 70% training, 30% validation
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)

    shape = np.shape(im)
    segmented = np.ones((1008,1920))*False
    for j in range(0, len(im[0])-patchSize-1, patchSize):
        for i in range(0, len(im)-patchSize-1, patchSize):
            patch = im[i:i+patchSize,j:j+patchSize]
            try:
                if estDeLaNeige(patch):
                    segmented[i:i+patchSize,j:j+patchSize] = True
            except:
                None
    return segmented

# fonction segmentation Unet
def segmentedUnet(img, mod_filename=f"{dir}/Modèles_trained/Segmentation_Unet/20ep_200step_V8.h5"):
    """
    entrée: image à segmenter, patchSize = taille du patch à classifier 
            tailles de patch dispo : 160
    sortie: masque de segmentation
    """
    def segmentedPatch(img, trained_model, patchSize = 160):
        """
        entrée: image à segmenter, patchSize = taille du patch à classifier 
        sortie: masque de segmentation
        """
        segmented = np.zeros((1008,1920))# *False
        for j in range(0, len(img[0])-patchSize-1, patchSize):
            for i in range(0, len(img)-patchSize-1, patchSize):
                patch = img[i:i+patchSize,j:j+patchSize]
                patch = patch[None, :, :, :]
                sortie = trained_model.predict(patch)
                res = np.squeeze(sortie)
                segmented[i:i+patchSize,j:j+patchSize] = np.where(res>0.1, 1, 0)
        return segmented

    input_shape = (160, 160, 3)
    mod = custom_unet(
        input_shape,
        filters=32,
        use_batch_norm=True,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        num_layers=5
    )
    mod.load_weights(mod_filename)
    segmented = segmentedPatch(img, mod, patchSize = 160)

    return segmented.astype(bool)

# ----------------------------------------------------------------------------------------
# --------------------  2. fonctions d'évaluation des performances -----------------------
# 
#            entrée : masque de réalité terrain et masque de segmentation
#            sortie : entier indicateur de performance
#
# ----------------------------------------------------------------------------------------

def getF1Scores(masquesReels, estimations):
    """ 
    entrée : masques réalité terrain, masques estimations
    return : liste des F1 score
    """
    scores = []
    for i in range(len(masquesReels)):
        scores.append(f1_score(masquesReels[i], estimations[i], average='samples'))
    return scores
    
# ----------------------------------------------------------------------------------------
# --------------------                    3. Autre                 -----------------------
# 
#            entrée : divers
#            sortie : None ou divers
#
# ----------------------------------------------------------------------------------------

# création d'une liste des masques de réalité terrain
def getMasks(imgs, imgs_path, shape_df_realite):
    """ 
    entrée : images, amages paths et masque réalité terrain
    return : liste des masques dans le même ordre que celui de la liste imgs
    """

    masques = []
    masque_names = []
    for row in range(len(shape_df_realite)):
        i = 0
        masks = []
        masque_names.append(shape_df_realite["filename"][row])
        while 1:
            try:
                # list_shape nombre du label de l'image row
                list_shape = str(i)
                x_poly = shape_df_realite['regions'][row][list_shape]["shape_attributes"]["all_points_x"]
                y_poly = shape_df_realite['regions'][row][list_shape]["shape_attributes"]["all_points_y"]
                polygon = [(x_poly[i],y_poly[i]) for i in range(len(x_poly))]

                (ny, nx) = imgs[0].shape[:2]
                x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x,y)).T

                path = Path(polygon)
                mask = path.contains_points(points)
                mask = mask.reshape((ny,nx))
                masks.append(mask)
                i += 1
            except KeyError:
                print(i, "sous masques trouvés")
                break

        if len(masks)!=1:
            for i in range(len(masks)):
                mask = mask | masks[i]
        masques.append(mask)

    listeMasques =[]
    numImagesAvecMasque = []

    for name in masque_names:
        i = 0
        while imgs_path[i] != name:
            i+=1 
        numImagesAvecMasque.append(i)  

    k = 0
    for i in range(len(imgs)):    
        if i in numImagesAvecMasque:
            listeMasques.append(masques[k].astype(bool))
            k += 1
        else:
            listeMasques.append(np.ones((1008, 1920), dtype=int).astype(bool))

    return listeMasques

# ratio de pixels de neige/pixels de pas neige
def getRatio(image, kPixelsNeige):
    """
    image: image original
    kPixelNeige: nombre de pixels trouvés suite à la segmentation

    return ratio: pourcentage de neige présene dans l'image
    """
    return 100*kPixelsNeige/(np.shape(image)[0]*np.shape(image)[1])


def equalize(src):
    """réalise une equalisation d'histogramme"""
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv.cvtColor(src, cv.COLOR_RGB2YCrCb)
    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv.equalizeHist(ycrcb_img[:, :, 0])
    # convert back to RGB color-space from YCrCb
    equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2RGB)
    return equalized_img 

def normalize(values):
    """retourne la liste en pourcentage"""
    res, somme = [], sum(values)
    for value in values:
        res.append(value*100/(somme))
    return res

def image_to_pandas(image):
    """ met en fomre les données pour le kmeans"""
    df = pd.DataFrame([image[:,:,0].flatten(),
                    image[:,:,1].flatten(),
                    image[:,:,2].flatten()]).T
    df.columns = ['Red_Channel','Green_Channel','Blue_Channel']
    return df


def plots(imgs, imgs_path, masques, estimation, scores):
    """
    imgs = liste des images testées
    imgs_path = pour la légende
    masques = masques de réalité terrain
    estimation = masque résultat de la segmentation
    score = ROC, F1_score
    """
    plt.rcParams.update({'font.size': 16})
    if scores == None:
        fig, axs = plt.subplots(3, len(imgs), figsize = (40, 13))
    else:
        fig, axs = plt.subplots(4, len(imgs), figsize = (40, 13))

    for i in range(len(imgs)):
        titre = imgs_path[i][:-29]

        axs[0][i].imshow(imgs[i])
        axs[0][i].set_title(titre)
        axs[0][i].set_axis_off()

        axs[1][i].imshow(masques[i])
        axs[1][i].set_axis_off()

        axs[2][i].imshow(estimation[i])
        axs[2][i].set_axis_off()

        if scores != None:
            axs[3][i].text(0.5, 0.5, 'F1 score\n'+str(round(scores[i], 3)), bbox={'facecolor':'cornflowerblue', 'alpha':0.5, 'pad':10},
            fontsize=17, ha='center')
            axs[3][i].set_axis_off()
    plt.show()
    return None