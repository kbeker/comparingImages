import sys
import random
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
import pandas as pd
from load_check_save_data import load_image, check, write_in_file

def init_variables():
    counter = 0
    ssim_norm_counter = 0
    ssim_canny_counter = 0
    mse_counter = 0
    good_crop = 0
    maybe_good_crop = 0
    nope_counter = 0
    lp = []
    ssim = []
    ssim_canny = [] 
    mse = []
    corr = []
    cord = []
    return counter, ssim_norm_counter, ssim_canny_counter, mse_counter, good_crop, maybe_good_crop, nope_counter, lp, ssim, ssim_canny, mse, corr, cord

def check_all_pixels(img1, img2, sizeH, sizeW, documentName):
    beginning = cv2.getTickCount()
    imageA, imageB, imageA_canny, imageB_canny, imgAH, imgAW, imgBH, imgBW = load_image(img1, img2)
    counter, ssim_norm_counter, ssim_canny_counter, mse_counter, good_crop, maybe_good_crop, nope_counter, lp, ssim, ssim_canny, mse, corr, cord = init_variables()
    for H in range(imgAH - sizeH):
        for W in range(imgAW - sizeW):
            ssim, ssim_canny, mse, corr, cord, lp, counter, good_crop, maybe_good_crop, nope_counter, time = check(imageA, imageA_canny, imageB, imageB_canny, H, sizeH, W, sizeW, good_crop, maybe_good_crop, nope_counter, counter, lp, ssim, ssim_canny, mse, corr, cord, beginning)
    write_in_file(ssim, ssim_canny, mse, corr, cord, imgAH, imgAW, lp, sizeH, sizeW, documentName)

check_all_pixels("bucket_800x800/bucket.png", "bucket_800x800/bucket1.png", 777, 777, "nowy_dokument")