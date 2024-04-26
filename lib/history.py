#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import json


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class SourceHistory:
    
    '''
    Helper used to save/load previous video source inputs,
    so user doesn't have to repeatedly enter inputs
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
