#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import os.path as osp
import json
from collections import OrderedDict

#---------------------------------------------------------------------------------------------------------------------
#%% Classes

class SourceHistory:
    
    '''
    Helper used to save/load previous video source inputs,
    so the user doesn't have to repeatedly enter inputs
    (especially helpful for rtsp urls)
    '''
    
    def __init__(self, save_key = "video_source", file_path = ".source_history.json"):
        
        self._save_key = save_key
        self._file_path = file_path
    
    def load(self):
        
        try:
            with open(self._file_path, "r") as in_file:
                history_data_dict = json.load(in_file)
            saved_source = history_data_dict.get(self._save_key, None)
            
        except (FileNotFoundError, AttributeError):
            # Fails if there is no file to load or if the file contains non-dict data
            saved_source = None
        
        return saved_source
    
    def save(self, video_source):
        
        try:
            save_data = {self._save_key: video_source}
            with open(self._file_path, "w") as out_file:
                json.dump(save_data, out_file, indent=2)
        
        except Exception as err:
            # Not expecting errors, but we don't want history save to cause full crashes
            print("",
                  "Unknown error saving video source history!",
                  "", str(err), sep = "\n", flush = True)
        
        return self


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def get_first_dict_item(dictionary: dict):
    return next(iter(dictionary.items()))

def get_file_to_path_lut(folder_path, allowable_exts=None) -> dict:
    
    '''
    Helper used to get a lookup between file names (without extensions)
    and their corresponding paths on the file system, sorted by file size
    Can also take a list/set of allowable_exts, which can be used to ignore
    all but specific target file types
    
    Returns a dictionary of file names to file paths, for example:
        {
            "file_1": "/path/to/file_1.jpg",
            "file2": "/some/other/path/file2.txt",
            "anotherFile": "/yet/another/path/to/anotherFile.pth",
            ...etc...
        }
    '''
    
    # Make sure allowable exts is set up correctly (should be iterable and each entry starts with '.')
    if isinstance(allowable_exts, str):
        allowable_exts = tuple(allowable_exts)
    allowable_exts = tuple(ext if ext.startswith(".") else f".{ext}" for ext in allowable_exts)
    
    # Get listing of yolo model files available
    files_list = os.listdir(folder_path)
    file_paths_list = [osp.join(folder_path, file) for file in files_list]
    
    # Keep only allowable exts, if given
    if allowable_exts is not None:
        is_allowed_ext = lambda file: osp.splitext(file.lower())[1] in allowable_exts
        file_paths_list = [path for path in file_paths_list if is_allowed_ext(path)]
    
    # Group file names with paths for output
    paths_smallest_first_list = sorted(file_paths_list, key=osp.getsize)
    name_to_path_dict = OrderedDict()
    for file_path in paths_smallest_first_list:
        file_name, file_ext = osp.splitext(osp.basename(file_path))
        name_to_path_dict[file_name] = file_path
    
    return name_to_path_dict
