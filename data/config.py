# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

def change_cfg_for_ssd512(cfg):
    cfg['min_dim'] = 512
    cfg['steps'] = [8, 16, 32, 64, 128, 256, 512]
    cfg['feature_maps'] = [64, 32, 16, 8, 4, 2, 1]
    cfg['min_sizes'] = [35.8, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8]
    cfg['max_sizes'] = [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6]
    cfg['aspect_ratios']= [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
    return cfg

def change_cfg_for_ssd512x640(cfg):
    cfg['min_dim'] = 512
    cfg['max_dim'] = 640
    cfg['steps'] = [8, 16, 32, 64, 128]
    cfg['feature_maps'] = [(64,80), (32,40), (16,20), (8,10), (4,5)]
    cfg['min_sizes'] = [18, 35.84, 76.8, 153.6, 230.4] # height
    cfg['max_sizes'] = [35.84, 76.8, 153.6, 230.4, 307.2] # height
    cfg['aspect_ratios']= [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    return cfg
    
# def change_cfg_for_ssd512x640(cfg):
#     cfg['min_dim'] = 512
#     cfg['max_dim'] = 640
#     cfg['steps'] = [4, 8, 16, 32, 64]
#     cfg['feature_maps'] = [(128,160), (64,80), (32,40), (16,20), (8,10)]
#     cfg['min_sizes'] = [15, 35.84, 76.8, 153.6, 230.4] # height
#     cfg['max_sizes'] = [35.84, 76.8, 153.6, 230.4, 307.2] # height
#     cfg['aspect_ratios']= [[2], [2, 3], [2, 3], [2, 3], [2, 3]]
#     return cfg

carplate = {
    'num_classes': 2,
    'lr_steps': (20000, 40000, 60000),
    'max_iter': 60000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
    'max_sizes': [60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'CARPLATE',
}

def change_cfg_for_ssd512_mobilenet(cfg):
    cfg['min_dim'] = 512
    cfg['steps'] = [16, 32, 64, 128, 256, 512]
    cfg['feature_maps'] = [32, 16, 8, 4, 2, 1]
    cfg['min_sizes'] = [35.84, 76.8, 168.96, 261.12, 353.28, 445.44]
    cfg['max_sizes'] = [76.8, 168.96, 261.12, 353.28, 445.44, 537.6]
    cfg['aspect_ratios']= [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    return cfg


two_stage_end2end = {
    'num_classes': 2,
    'lr_steps': (20000, 40000, 60000),
    'max_iter': 60000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'feature_maps_2': [56, 28, 14],
    'min_dim': 300,
    'min_dim_2': 56,
    'expand_num': 3,
    'steps': [8, 16, 32, 64, 100, 300],
    'steps_2': [1, 2, 4],
    'min_sizes': [30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
    'max_sizes': [60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
    'min_sizes_2': [5.6, 11.2, 50.4],
    'max_sizes_2': [11.2, 50.4, 89.6],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_ratios_2': [[2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'TWO_STAGE_END2END',
}
