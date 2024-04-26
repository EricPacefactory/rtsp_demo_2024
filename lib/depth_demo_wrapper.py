#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2
import numpy as np
import onnxruntime

from lib.downloading import download_missing_model_files
from lib.misc import get_first_dict_item, get_file_to_path_lut


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class DepthDemo:
    
    # Hard-coded configuration for depth-anything models
    _proc_wh = (518, 518)
    _mean_rgb = np.float32([0.485, 0.456, 0.406])
    _std_rgb = np.float32([0.229, 0.224, 0.225])
    
    # For reference, download links to model files
    _download_urls = [
        "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v1.0.0/depth_anything_vits14.onnx",
        "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v1.0.0/depth_anything_vitb14.onnx",
    ]
    
    def __init__(self, models_folder_path = "models/depth"):
        
        # Get model files if needed
        download_missing_model_files(self._download_urls, models_folder_path)
        
        self._name_to_model_dict = self._load_models(models_folder_path)
        self._num_models = len(self._name_to_model_dict)
        self._model_select, _ = get_first_dict_item(self._name_to_model_dict)
    
    def get_model_names(self) -> list[str]:
        return list(self._name_to_model_dict.keys())
    
    def set_model_select(self, model_select: str):
        self._model_select = model_select
    
    def process_frame(self, frame_bgr):
        
        ort_session = self._name_to_model_dict[self._model_select]
        
        # Image must be RGB ordered, with CxHxW shape, with normalized mean/standard deviation
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        scaled_frame = cv2.resize(frame_rgb, dsize=self._proc_wh)
        scaled_frame = (np.float32(scaled_frame)/255.0 - self._mean_rgb) / self._std_rgb
        scaled_frame = np.transpose(scaled_frame, (2, 0, 1))
        scaled_frame = np.expand_dims(scaled_frame, axis=0)
        
        depth_result = ort_session.run(None, {"image": scaled_frame})[0].squeeze()
        
        return depth_result
    
    def draw_results(self, depth_result_1ch, display_shape, use_high_contrast = True):
        
        disp_h, disp_w = display_shape[0:2]
        
        depth_min, depth_max = depth_result_1ch.min(), depth_result_1ch.max()
        depth_norm = (depth_result_1ch - depth_min) / (depth_max - depth_min)
        depth_uint8 = np.uint8(255 * depth_norm)
        if use_high_contrast:
            depth_uint8 = cv2.equalizeHist(depth_uint8)
        
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        depth_color = cv2.resize(depth_color, dsize=(disp_w, disp_h))
        
        return depth_color
    
    def _load_models(self, folder_path):
        
        '''
        Helper which loads multiple depth models, smallest first
        Returns a dictionary whose keys are the model names (no file extension) and
        have corresponding values of the models (onnx sessions) themselves
        '''
        
        # Helper used to create onnx sessions
        make_ort = lambda path: onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        
        name_to_path_dict = get_file_to_path_lut(folder_path, allowable_exts = [".onnx"])
        name_to_model_dict = {name: make_ort(path) for name, path in name_to_path_dict.items()}
        
        return name_to_model_dict
