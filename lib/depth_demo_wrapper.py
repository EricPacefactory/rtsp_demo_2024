#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os

import cv2
import numpy as np
import onnxruntime


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class DepthDemo:
    
    _mean_rgb = np.float32([0.485, 0.456, 0.406])
    _std_rgb = np.float32([0.229, 0.224, 0.225])
    
    def __init__(self, models_folder_path):
        
        models_list, proc_wh_list = self._load_models(models_folder_path)
        self._orts_list = models_list
        self._process_wh_list = proc_wh_list
        self._idx_select = 0
        self._num_models = len(self._orts_list)
    
    def set_model_select(self, selection_index: int):
        self._idx_select = selection_index
    
    def process_frame(self, frame_bgr):
        
        idx = self._idx_select % self._num_models
        ort_session = self._orts_list[idx]
        proc_wh = self._process_wh_list[idx]
        
        # Image must be RGB ordered, with CxHxW shape, with normalized mean/standard deviation
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        scaled_frame = cv2.resize(frame_rgb, dsize=proc_wh)
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
        
        ''' Helper which loads multiple models, smallest first '''
        
        # For clarity
        allowable_exts = (".onnx",)
        is_allowed_ext = lambda file: os.path.splitext(file.lower())[1] in allowable_exts
        ort_providers = ["CPUExecutionProvider"]
        
        # Get listing of model files available
        files_list = os.listdir(folder_path)
        file_paths_list = [os.path.join(folder_path, file) for file in files_list]
        model_paths_list = [path for path in file_paths_list if is_allowed_ext(path)]
        paths_smallest_first_list = sorted(model_paths_list, key=os.path.getsize)
        
        # Build models + processing sizes per model
        models_list = []
        proc_wh_list = []
        for path in paths_smallest_first_list:
            model = onnxruntime.InferenceSession(path, providers=ort_providers)
            models_list.append(model)
            proc_wh_list.append((518,518))
        
        # Add extra copy of biggest model with larger processing size, out of curiosity
        models_list.append(models_list[-1])
        proc_wh_list.append((756,756))
        
        return models_list, proc_wh_list

