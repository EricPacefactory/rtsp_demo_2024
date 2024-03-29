#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2

# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class DisplayWindow:
    
    _quit_keycodes = (ord("q"), 27)
    
    def __init__(self, window_title, create_on_init = True):
        
        self._name = window_title
        if create_on_init:
            self.create_window()
        
        # Attach object that supports running multiple (independent) callbacks
        self._cbs = CallbackSequencer()
        cv2.setMouseCallback(self._name, self._cbs)
    
    def create_window(self):
        cv2.namedWindow(self._name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        return self
    
    def imshow(self, display_frame):
        
        cv2.imshow(self._name, display_frame)
        keypress = cv2.waitKey(1) & 0xFF
        req_close = keypress in self._quit_keycodes
        
        return req_close, keypress
    
    def add_callbacks(self, *callbacks):
        self._cbs.add(*callbacks)
        return self
    
    def close(self):
        cv2.destroyWindow(self._name)
        return self
    
    def close_all(self):
        cv2.destroyAllWindows()
        return self




class CallbackSequencer:
    
    '''
    Simple wrapper used to execute more than one callback on a single opencv window
    
    Example usage:
        
        # Set up window that will hold callbacks
        winname = "Display"
        cv2.namedWindow(winname)
        
        # Attach sequencer to window
        cb_seq = CallbackSequence(cb_1, cb_2)
        cv2.setMouseCallback(winname, cb_seq)
        
        #  Add callbacks to sequencer
        cb_1 = MakeCB(...)
        cb_2 = MakeCB(...)
        cb_seq.add(cb_1)
        cb_seq.add(cb_2)
        # or cb_seq.add(cb_1, cb_2)
    '''
    
    def __init__(self, *callbacks):
        self._callbacks = [cb for cb in callbacks]
    
    def add(self, *callbacks):
        self._callbacks.extend(callbacks)
    
    def __call__(self, event, x, y, flags, param) -> None:
        for cb in self._callbacks:
            cb(event, x, y, flags, param)
        return
    
    def __getitem__(self, index):
        return self._callbacks[index]
    
    def __iter__(self):
        yield from self._callbacks