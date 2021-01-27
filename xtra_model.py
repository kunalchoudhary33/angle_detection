import tensorflow as tf
import cv2
import time
import argparse
import numpy as np
import base64 
import logging
from scipy.ndimage.filters import gaussian_filter

import util
from config_reader import config_reader
from model import get_testing_model

logger = logging.getLogger('run')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class xtra_model:
    def __init__(self):
        self.model = get_testing_model()
        self.model.load_weights('./model/keras/model.h5')
        logger.info('Model loaded successfully')
        
    
    def process (self, input_image, params, model_params):
        ''' Start of finding the Key points of full body using Open Pose.'''
        oriImg = input_image # B,G,R order  
        multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        for m in range(1):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                            model_params['padValue'])
            input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
            output_blobs = self.model.predict(input_img)
            heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
            heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                                interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                    :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
            paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                            interpolation=cv2.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
            paf_avg = paf_avg + paf / len(multiplier)

        all_peaks = [] #To store all the key points which a re detected.
        peak_counter = 0
        

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        connection_all = []
        special_k = []
        mid_num = 10


        #canvas = frame# B,G,R order
        keypoints = {}
        for i in range(18): #drawing all the detected key points.
            for j in range(len(all_peaks[i])):
                
                if(i==1):
                    keypoints['Neck'] = all_peaks[i][j][0:2]
                if(i==4):
                    keypoints['Rwri'] = all_peaks[i][j][0:2]
                if(i==7):
                    keypoints['Lwri'] = all_peaks[i][j][0:2]
                

        return keypoints

    def get_feedback(self, source_img):
        
        # encoded_data = s_image.split(',')[1]
        # nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        # source_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #logger.info("Image read")

        params, model_params = config_reader()
        keypoints = xtra_model.process(self, source_img, params, model_params)
        #feedback = rule(keypoints)
        #logger.info(keypoints)
        #logger.info(feedback)
        return keypoints