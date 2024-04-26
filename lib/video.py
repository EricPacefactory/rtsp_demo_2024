#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os.path as osp
import cv2
import numpy as np
from time import perf_counter

# Typing
from typing import Protocol
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class FrameReader(Protocol):
    
    ''' Defines standard interface for all types of frame-by-frame readers (e.g rtsp/webcam/files) '''
    
    def __iter__(self): ...
    def __next__(self) -> ndarray | None: ...
    def get_shape(self) -> tuple[int,int,int]: ...
    def read(self) -> tuple[bool, ndarray | None]: ...
    def release(self) -> None: ...
    def exhaust_buffered_frames(self, max_frames_to_exhaust: int) -> None: ...


class VideoStreamReader(FrameReader):
    
    '''
    Class for reading from 'streaming' video sources (e.g. rtsp or webcams)
    Includes support for skipping frames that read too slowly
    '''
    
    # .................................................................................................................
    
    def __init__(self, video_source: str | int, min_read_time_ms = 10):
        
        self._source = video_source
        self._min_read_time_ms = min_read_time_ms
        self.cap = cv2.VideoCapture(self._source)
        if not self.cap.isOpened():
            raise SystemExit("Unable to open video source!")
        
        frame_w = int(round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        frame_h = int(round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._shape = (frame_h, frame_w, 3)
    
    # .................................................................................................................
    
    def __iter__(self):
        
        ''' Called when using this object in an iterator (e.g. for loops) '''
        
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self._source)
        
        # Get rid of frames which have built up since object instantiation
        self.exhaust_buffered_frames()
        
        return self
    
    # .................................................................................................................

    def __next__(self) -> ndarray | None:

        '''
        Iterator that provides frame data from a video capture object
        Returns frame_bgr
        
        Note: This allows instances of this class to be used in for loops
        '''
        
        # Read next frame, or loop back to beginning if there are no more frames
        read_ok, frame_bgr = self.read()
        if not read_ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            read_ok, frame_bgr = self.read()
            if not read_ok: raise IOError("Error reading frames! Disconnected?")
        
        return frame_bgr
    
    # .................................................................................................................
    
    def get_shape(self) -> tuple[int,int,int]:
        return self._shape
    
    # .................................................................................................................
    
    def get_playback_position(self) -> float:
        return 0.0
    
    # .................................................................................................................
    
    def read(self) -> tuple[bool, ndarray | None]:
        
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

    def exhaust_buffered_frames(self, max_frames_to_exhaust = 300) -> bool:
        
        '''
        Helper used to 'exhaust' buffered frames
        Works by reading frames as quickly as possible until it takes
        'a long time' to read the next frame.
        
        The idea being that buffered frames read very quickly since they
        are immediately available, whereas non-buffered frames need to wait
        for 'real life' to progress (and therefore come in slower).
        
        Returns False if there were no buffered frames.
        '''
        
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        read_ms_threshold = round(0.85 * 1000.0 / video_fps)
        
        for N in range(max_frames_to_exhaust):
            
            t1 = perf_counter()
            rec_frame = self.cap.grab()
            t2 = perf_counter()
            assert rec_frame, "Error exhausting frames, no data!"
            
            read_time_ms = round(1000 * (t2 - t1))
            long_read_time = (read_time_ms >= read_ms_threshold)
            if long_read_time:
                break
        
        had_buffered_frames = N > 0
        return had_buffered_frames
    
    # .................................................................................................................
    
    def release(self) -> None:
        self.cap.release()
        return
    
    # .................................................................................................................


class VideoFileReader(VideoStreamReader):
    
    ''' Class used to read frames from a video file, as if it were a stream '''
    
    # .................................................................................................................
    
    def __init__(self, video_path: str):
        super().__init__(video_path)
        self._total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # .................................................................................................................
    
    def get_playback_position(self) -> float:
        return min(1.0, self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self._total_frames)
    
    # .................................................................................................................
    
    def set_playback_position(self, position_norm_01):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, round(self._total_frames * position_norm_01))
        return self
    
    # .................................................................................................................
    
    def read(self):
        return self.cap.read()
    
    # .................................................................................................................
    
    def exhaust_buffered_frames(self, max_frames_to_exhaust = 300):
        # No need to exhaust frames on files
        return True
    
    # .................................................................................................................
    
    @staticmethod
    def is_valid_video_file(video_source: str) -> bool:
        
        '''
        Helper used to check if the given video source is a valid video file
        This can be used to exclude valid files that aren't videos (e.g .jpg for example)
        '''
        
        # First check that we're dealing with a valid file
        is_valid = osp.exists(video_source)
        if not is_valid:
            return is_valid
        
        # Check that we can open given file as a video
        vcap = cv2.VideoCapture(video_source)
        is_valid = vcap.isOpened()
        vcap.release()
        
        return is_valid
    
    # .................................................................................................................


class ImageFileReader(FrameReader):
    
    '''
    Class used as a stand-in for loading images as if they were video files
    (which just repeat the same frame over and over)
    '''
    
    # .................................................................................................................
    
    def __init__(self, image_file_path: str):
        
        self._source = image_file_path
        
        # Make sure we got a valid file before loading
        assert osp.exists(image_file_path), f"Invalid file path: {image_file_path}"
        self._image_path = image_file_path
        self._image = cv2.imread(self._image_path)
        
        # Make sure we can interpret the loaded file as an image
        assert (self._image is not None), f"Ivalid image file: {image_file_path}"
        self._img_h, self._img_w = self._image.shape[0:2]
        
        # Mimic other video readers
        self._shape = self._image.shape
    
    # .................................................................................................................
    
    def __iter__(self): return self
    
    def __next__(self): return self.read()
    
    def get_shape(self): return self._shape
    
    def read(self): return self._image.copy()
    
    def release(self): return
    
    def exhaust_buffered_frames(self, max_frames_to_exhaust = 300): return True
    
    # .................................................................................................................
    
    @staticmethod
    def is_valid_image_file(video_source: str) -> bool:
        
        '''
        Helper used to check if the given video source is a valid video file
        This can be used to exclude valid files that aren't videos (e.g .jpg for example)
        '''
        
        # First check that we're dealing with a valid file
        is_valid = osp.exists(video_source)
        if not is_valid:
            return is_valid
        
        # Check that we can open given file as an image
        image = cv2.imread(video_source)
        is_valid = image is not None
        
        return is_valid
    
    # .................................................................................................................


class PlaybackBar:
    
    def __init__(self, frame_reader_ref, bg_color = (100,95,85), line_color = (255,255,255), bar_height = 80):
        
        # Store access to frame reader, so we can control playback
        self._reader = frame_reader_ref
        
        # Figure out whether we should be enabled, based on whether we can control the reader
        is_controllable_source = False
        try:
            pos = self._reader.get_playback_position()
            self._reader.set_playback_position(pos)
            is_controllable_source = True
        except AttributeError:
            is_controllable_source = False
        
        # Graphics
        self.height_px = bar_height
        self._bg_color = bg_color
        self._line_color = line_color
        self._line_thickness = 2
        self._base_img = np.full((1,1,3), 0, dtype = np.uint8)
        
        # Interaction settings
        self._mouse_pressed = False
        self._x_norm = 0
        self._frame_w = 0
        self._interact_y_offset = 0
        self._interact_y1y2 = (-10, -10)
        self._enable = is_controllable_source
    
    def enable(self, enable = True):
        self._enable = enable
        return self

    def set_y_offset(self, y_offset_px):
        self._interact_y_offset = y_offset_px
        return self

    def __call__(self, event, x, y, flags, param) -> None:
        
        # Don't run callback when disabled
        if not self._enable:
            return
        
        # Keep track of relative position for playback control (when active)
        # -> We do this here (instead of only on mouse down) so that the user can
        #    continue to control playback while dragging the mouse, even if they
        #    drag 'outside' the playback bar area (as long as they began the drag inside it!)
        bar_w = self._base_img.shape[1]
        self._x_norm = min(1.0, max(0.0, x / (bar_w - 1)))
        
        # Respond to mouse release, since this should stop playback control
        if event == cv2.EVENT_LBUTTONUP:
            self._mouse_pressed = False
            return
        
        # Keep track of when the mouse is down (playback control is only active in this state)
        y1, y2 = self._interact_y1y2
        is_interacting_y = y1 < (y - self._interact_y_offset) < y2
        mouse_is_down = event == cv2.EVENT_LBUTTONDOWN
        if is_interacting_y and mouse_is_down:
            self._mouse_pressed = True
        
        return
    
    def draw_bar(self, frame_w):
        
        # Update the base image sizing if needed
        if frame_w != self._base_img.shape[1]:
            self._base_img = np.full((self.height_px, frame_w, 3), self._bg_color, dtype=np.uint8)
        
        # Figure out where to draw the playback position indicator line
        playback_pos_norm = self._reader.get_playback_position()
        x_px = round((frame_w - 1) * playback_pos_norm)
        xy1 = (x_px, -5)
        xy2 = (x_px, self._base_img.shape[0] + 5)
        
        # Draw the indicator line
        bar_img = self._base_img.copy()
        bar_img = cv2.line(bar_img, xy1, xy2, self._line_color, self._line_thickness)
        
        return bar_img
    
    def append_to_frame(self, frame):
        
        if not self._enable:
            return frame
        
        frame_h, frame_w = frame.shape[0:2]
        bar_img = self.draw_bar(frame_w)
        self._interact_y1y2 = (frame_h, frame_h + bar_img.shape[0])
        
        return np.vstack((frame, bar_img))

    def adjust_playback_on_drag(self):
        
        # Only adjust position when the mouse is held down
        if self._mouse_pressed:
            self._reader.set_playback_position(self._x_norm)
        
        return self

# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def make_video_reader(video_source):
    
    ''' Helper used to instantiate a video reader, based on the input type (eg. rtsp vs. video file) '''
    
    # Make sure we don't get extra spaces & look for possible webcam inputs
    video_source = str(video_source).strip()
    if video_source.lower() in ("webcam", "cam"):
        video_source = "0"
    
    # Try to load as an image first, since this is 'least intrusive'
    is_image = ImageFileReader.is_valid_image_file(video_source)
    if is_image:
        return "image", ImageFileReader(video_source)
    
    # Try video files
    is_video_file = VideoFileReader.is_valid_video_file(video_source)
    if is_video_file:
        return "video", VideoFileReader(video_source)
    
    # Check if source is an integer (implies webcam)
    is_webcam = video_source.isnumeric()
    if is_webcam:
        return "webcam", VideoStreamReader(int(video_source))
    
    # If we get here, assume we got an rtsp source which is otherwise hard to verify!
    return "rtsp", VideoStreamReader(video_source)