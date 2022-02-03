import torch
import os
from ssd import build_ssd

def export():
    weight_path = '/workspace/LPD-end-to-end/CCPD_carplate_bbox_weights_12345789_1ssd512_25000.pth'    
    detector_onnx_save_path = '/workspace/LPD-end-to-end/CCPD_carplate_bbox_weights_12345789_1ssd512_25000.onnx'
    net = build_ssd('export', 512, 2)
    net.load_weights(weight_path)
    dummy_input = torch.rand((1, 3, 512, 512))
    torch.onnx.export(net, dummy_input, detector_onnx_save_path, input_names=['input'],
                              output_names=['output'])

if __name__ == '__main__':
    export()