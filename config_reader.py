#from configobj import ConfigObj
import numpy as np


def config_reader():
    # config = ConfigObj('config')

    # param = config['param']
    # model_id = param['modelID']
    # model = config['models'][model_id]
    # model['boxsize'] = int(model['boxsize'])
    # model['stride'] = int(model['stride'])
    # model['padValue'] = int(model['padValue'])
    # #param['starting_range'] = float(param['starting_range'])
    # #param['ending_range'] = float(param['ending_range'])
    # param['octave'] = int(param['octave'])
    # param['use_gpu'] = int(param['use_gpu'])
    # param['starting_range'] = float(param['starting_range'])
    # param['ending_range'] = float(param['ending_range'])
    # param['scale_search'] = map(float, param['scale_search'])
    # param['thre1'] = float(param['thre1'])
    # param['thre2'] = float(param['thre2'])
    # param['thre3'] = float(param['thre3'])
    # param['mid_num'] = int(param['mid_num'])
    # param['min_num'] = int(param['min_num'])
    # param['crop_ratio'] = float(param['crop_ratio'])
    # param['bbox_ratio'] = float(param['bbox_ratio'])
    # param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])


    param = {}
    model_id = 1
    model = {}
    model['boxsize'] = int(368)
    model['stride'] = int(8)
    model['padValue'] = int(128)
    #param['starting_range'] = float(param['starting_range'])
    #param['ending_range'] = float(param['ending_range'])
    param['octave'] = int(3)
    param['use_gpu'] = int(1)
    param['starting_range'] = float(0.8)
    param['ending_range'] = float(2)
    param['scale_search'] = map(float, (0.5, 1, 1.5, 2))
    param['thre1'] = float(0.1)
    param['thre2'] = float(0.05)
    param['thre3'] = float(0.5)
    param['mid_num'] = int(10)
    param['min_num'] = int(4)
    param['crop_ratio'] = float(2.5)
    param['bbox_ratio'] = float(0.25)
    param['GPUdeviceNumber'] = int(0)


    return param, model

if __name__ == "__main__":
    config_reader()
