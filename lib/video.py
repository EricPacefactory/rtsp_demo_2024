#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2
from time import perf_counter


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class VideoReader:
    
    # .................................................................................................................
    
    def __init__(self, video_source, min_read_time_ms = 10):
        
        self._source = video_source
        self._min_read_time_ms = min_read_time_ms
        self.cap = cv2.VideoCapture(self._source)
        
        frame_w = int(round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        frame_h = int(round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.shape = (frame_h, frame_w, 3)
    
    # .................................................................................................................
    
    def __iter__(self):
        
        ''' Called when using this object in an iterator (e.g. for loops) '''
        
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self._source)
        
        return self
    
    # .................................................................................................................

    def __next__(self):

        ''' Iterator that provides frame data from a video capture object. Returns frame_bgr '''
        
        # Read next frame, or loop back to beginning if there are no more frames
        read_ok, frame_bgr = self.read()
        if not read_ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            read_ok, frame_bgr = self.read()
            if not read_ok: raise IOError("Error reading frames! Disconnected?")
        
        return frame_bgr
    
    # .................................................................................................................
    
    def read(self):
        
        ''' Read frames from video, with frame skipping if read happens too fast '''
        
        # Only 'use' frames that take some time to read
        # -> This implies we 'waited' for the frame, rather than reading from a buffer
        while True:
            t1 = perf_counter()
            rec_frame = self.cap.grab()
            t2 = perf_counter()
            
            time_taken_ms = int(1000 * (t2 - t1))
            if (time_taken_ms > self._min_read_time_ms) or (not rec_frame):
                break
        
        # Try decoding frame data
        rec_frame, frame = self.cap.retrieve() if rec_frame else (rec_frame, None)
        
        return rec_frame, frame

    # .................................................................................................................

    def exhaust_buffered_frames(self, max_frames_to_exhaust = 300):
        
        ''' Helper used to 'exhaust' buffered frames '''
        
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        read_ms_threshold = round(0.85 * 1000.0 / video_fps)
        
        finished_exhaust = False
        for N in range(max_frames_to_exhaust):
            
            t1 = perf_counter()
            rec_frame = self.cap.grab()
            t2 = perf_counter()
            assert rec_frame, "Error exhausting frames, no data!"
            
            read_time_ms = round(1000 * (t2 - t1))
            finished_exhaust = (read_time_ms >= read_ms_threshold)
            if finished_exhaust:
                break
        
        return finished_exhaust
    
    # .................................................................................................................
    
    def release(self):
        return self.cap.release()
    
    # .................................................................................................................


class FileReader(VideoReader):
    
    def __init__(self, video_path):
        
        # Inherit from parent
        super().__init__(video_path)
    
    # .................................................................................................................
    
    def read(self):
        return self.cap.read()
    
    # .................................................................................................................
    
    def exhaust_buffered_frames(self, max_frames_to_exhaust = 300):
        return True
    
    # .................................................................................................................


"""
class WebcamReader(RTSPReader):
    
    def __init__(self, webcam_select: int = 0):
        
        self._source = webcam_select
        self.cap = None
        self.shape = None
        self.open()
    
    def verify_source(self, video_source):
        
        ok_source = False
        try:
            webcam_select = int(video_source)
            ok_source = True
            
        except ValueError:
            pass
        
        return ok_source, webcam_select

class RTSPReader:
    
    def __init__(self, webcam_select: int = 0):
        
        self._source = webcam_select
        self.cap = None
        self.shape = None
        self.open()
    
    def verify_source(self, video_source):
        
        return
    
    def open(self):
        
        if (self.cap is None) or (not self.cap.isOpened()):
            self.cap = cv2.VideoCapture(self._source)
            frame_w = int(round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            frame_h = int(round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.shape = (frame_h, frame_w, 3)
        
        return self.cap.isOpened()
    
    def read(self):
        
        ''' Read frames from video, with frame skipping if read happens too fast '''
        
        # Only 'use' frames that take some time to read
        # -> This implies we 'waited' for the frame, rather than reading from a buffer
        while True:
            t1 = perf_counter()
            rec_frame = self.cap.grab()
            t2 = perf_counter()
            
            time_taken_ms = int(1000 * (t2 - t1))
            if (time_taken_ms > self._min_read_time_ms) or (not rec_frame):
                break
        
        # Try decoding frame data
        rec_frame, frame = self.cap.retrieve() if rec_frame else (rec_frame, None)
        
        return rec_frame, frame
    
    def exhaust_buffered_frames(self, max_frames_to_exhaust = 200):
        
        ''' Helper used to 'exhaust' buffered frames '''
        
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        read_ms_threshold = round(0.85 * 1000.0 / video_fps)
        
        finished_exhaust = False
        for N in range(max_frames_to_exhaust):
            
            t1 = perf_counter()
            rec_frame = self.cap.grab()
            t2 = perf_counter()
            assert rec_frame, "Error exhausting frames, no data!"
            
            read_time_ms = round(1000 * (t2 - t1))
            finished_exhaust = (read_time_ms >= read_ms_threshold)
            if finished_exhaust:
                break
        
        return finished_exhaust
"""