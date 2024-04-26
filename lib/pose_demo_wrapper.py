#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from ultralytics import YOLO

from lib.downloading import download_missing_model_files
from lib.misc import get_first_dict_item, get_file_to_path_lut


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class PoseDemo:
    
    # For reference, download links to model files
    _download_urls = [
        "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt",
    ]
    
    def __init__(self, models_folder_path = "models/pose"):
        
        # Get model files if needed
        download_missing_model_files(self._download_urls, models_folder_path)
        
        self._name_to_model_dict = self._load_models(models_folder_path)
        self._num_models = len(self._name_to_model_dict)
        self._model_select, _ = get_first_dict_item(self._name_to_model_dict)
    
    def get_model_names(self) -> list[str]:
        return list(self._name_to_model_dict.keys())
    
    def set_model_select(self, model_select_name: str):
        self._model_select = model_select_name
        return self
    
    def process_frame(self, frame):
        
        model = self._name_to_model_dict[self._model_select]
        pose_results = model(frame, verbose=False)
        
        return pose_results
    
    def draw_results(self, results, display_frame):
        
        for result in results:
            # boxes = result.boxes
            # keypoints = result.keypoints
            display_frame = result.plot(boxes = False, img=display_frame)
        
        return display_frame
    
    def _load_models(self, folder_path) -> dict:
        
        '''
        Helper which loads multiple yolo models, smallest first
        Returns a dictionary whose keys are the model names (no file extension) and
        have corresponding values of the models themselves
        '''
        
        # Get listing of yolo model files available
        name_to_paths_dict = get_file_to_path_lut(folder_path, allowable_exts = (".pt", ".pth"))
        name_to_model_dict = {name: YOLO(path).to("cpu") for name, path in name_to_paths_dict.items()}
        
        return name_to_model_dict
