#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os

from ultralytics import YOLO


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class PoseDemo:
    
    def __init__(self, model_folder_path):
        
        self._models_list = self._load_models(model_folder_path)
        self._idx_select = 0
        self._num_models = len(self._models_list)
    
    def set_model_select(self, selection_index: int):
        self._idx_select = selection_index
    
    def process_frame(self, frame):
        
        idx = self._idx_select % self._num_models
        model = self._models_list[idx]
        pose_results = model(frame, verbose=False)
        
        return pose_results
    
    def draw_results(self, results, display_frame):
        
        for result in results:
            # boxes = result.boxes
            # keypoints = result.keypoints
            display_frame = result.plot(boxes = False, img=display_frame)
        
        return display_frame
    
    def _load_models(self, folder_path):
        
        ''' Helper which loads multiple yolo models, smallest first '''
        
        # For clarity
        allowable_exts = (".pt", ".pth")
        is_allowed_ext = lambda file: os.path.splitext(file.lower())[1] in allowable_exts
        
        # Get listing of yolo model files available
        files_list = os.listdir(folder_path)
        file_paths_list = [os.path.join(folder_path, file) for file in files_list]
        model_paths_list = [path for path in file_paths_list if is_allowed_ext(path)]
        paths_smallest_first_list = sorted(model_paths_list, key=os.path.getsize)
        
        return [YOLO(path).to("cpu") for path in paths_smallest_first_list]
