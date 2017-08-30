import sys
import random
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
import pandas as pd
from compare_metrics import compare_histograms, compare_images, compare_images_canny

def load_image(image1, image2):
    imageA = cv2.imread(image1)
    imageB = cv2.imread(image2)
    (imgAH, imgAW) = imageA.shape[:2]
    (imgBH, imgBW) = imageB.shape[:2]
    imageA_canny = cv2.Canny(imageA, 0, 0)
    imageB_canny = cv2.Canny(imageB, 0, 0)
    return imageA, imageB, imageA_canny, imageB_canny, imgAH, imgAW, imgBH, imgBW

def check(imageA, imageA_canny, imageB, imageB_canny, 
    H, sizeH, W, sizeW, good_crop, maybe_good_crop, nope_counter, 
    counter, lp, ssim, ssim_canny, mse, corr, cord, beginning):
    
    cropWind = imageA[H:H + sizeH, W:W + sizeW]
    cropWind_canny = imageA_canny[H:H + sizeH, W:W + sizeW]
    cropWind2 = imageB[H:H + sizeH, W:W + sizeW]
    cropWind2_canny = imageB_canny[H:H + sizeH, W:W + sizeW]
    imgCorr = compare_histograms(cropWind, cropWind2)
    SSIM_normal, MSE_normal = compare_images(cropWind, cropWind2)
    SSIM_canny = compare_images_canny(cropWind_canny, cropWind2_canny)
    if SSIM_normal > 0.98 and SSIM_canny > 0.7 and MSE_normal < 50:
        good_crop += 1
    elif SSIM_normal > 0.9 and SSIM_canny > 0.5 and MSE_normal < 65:
        maybe_good_crop += 1
    else:
        nope_counter += 1
    next_image = cv2.getTickCount()
    time = (next_image - beginning)/ cv2.getTickFrequency()
    print("Crop Windows checked: %.0f Crop Windows OK: %.0f Crop Windows MAYBE: %.0f Crop Windows NOPE: %.0f. TIME: %.0f" % (counter+1, good_crop, maybe_good_crop, nope_counter, time))
    counter += 1
    lp.append(str(counter) + ".")
    ssim.append(SSIM_normal)
    ssim_canny.append(SSIM_canny)
    mse.append(MSE_normal)
    corr.append(imgCorr)
    cord.append(str(H) + ":" + str(W))
    return ssim, ssim_canny, mse, corr, cord, lp, counter, good_crop, maybe_good_crop, nope_counter, time

def write_in_file(ssim, ssim_canny, mse, corr, cord, height, width, counter, sizeH, sizeW, documentName):
    name_of_file = documentName + "_" + str(height) + "x" + str(width) + "_cropWinSize_" + str(sizeH) + "x" + str(sizeW) + ".xlsx"
    data = {'L.p':counter,
            'CORRDINATE':cord,
            'SSIM':ssim,
            'SSIM_canny':ssim_canny,
            'MSE':mse,
            'CORRELATION':corr}
    data = pd.DataFrame(data)
    data.set_index('L.p', inplace=True)
    data = [data]
    result = pd.concat(data, axis=1)
    result.to_excel(name_of_file)
