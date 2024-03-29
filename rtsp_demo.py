#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import argparse

import cv2
import numpy as np

from lib.display import DisplayWindow
from lib.video import VideoReader

from lib.aruco_demo_wrapper import ArucoDemo
from lib.pose_demo_wrapper import PoseDemo
from lib.depth_demo_wrapper import DepthDemo


# ---------------------------------------------------------------------------------------------------------------------
#%% Script args

# Define script arguments
parser = argparse.ArgumentParser(description="Demo script for running pose/ArUco/depth models on live video")
parser.add_argument("-i", "--rtsp_url", default=None, type=str,
                    help="Camera RTSP url to connect to, eg: rtsp://user:pass@192.168.0.15:554/video")
parser.add_argument("-s", "--display_size", default=None, type=int,
                    help="Set maximum side length for displayed image")
parser.add_argument("-w", "--use_webcam", default=False, action="store_true",
                    help="Toggle use of webcam, instead of RTSP source")

# For convenience
args = parser.parse_args()
arg_rtsp_url = args.rtsp_url
arg_display_size = args.display_size
arg_use_webcam = args.use_webcam


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class SelectionBar:
    
    def __init__(self, *titles, select_color = (0,255,0), bar_height = 60,
                 fg_color = (255,255,255), bg_color = (40,40,40), line_color = (80,80,80)):
        
        self._titles = list(titles)
        self.height_px = bar_height
        self._fg_color = fg_color
        self._bg_color = bg_color
        self._select_color = select_color
        self._line_color = line_color
        self._base_img = np.full((1,1,3), 0, dtype = np.uint8)
        
        
        self._interact_y_offset = 0
        self._interact_y1y2 = (-10, -10)
        self._idx_select = 0
        self._enable = True
    
    def __call__(self, event, x, y, flags, param) -> None:
        
        # Don't run callback when disabled
        if not self._enable:
            return
        
        # Only respond when mouse is over top of slider
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        # Bail if mouse is at wrong height
        y1, y2 = self._interact_y1y2
        is_interacting_y = y1 < (y - self._interact_y_offset) < y2
        if not is_interacting_y:
            return
        
        # Bail if mouse is at wrong x location
        bar_w = self._base_img.shape[1]
        is_interacting_x = 0 < x < bar_w
        if not is_interacting_x:
            return
        
        # Set selection to nearest title box
        x_norm = x / (bar_w - 1)
        num_titles = len(self._titles)
        idx_select = int(x_norm * num_titles)
        idx_select = min(num_titles - 1, idx_select)
        idx_select = max(0, idx_select)
        self._idx_select = idx_select
        
        return
    
    def enable(self, enable = True):
        self._enable = enable
        return self
    
    def set_y_offset(self, y_offset_px):
        self._interact_y_offset = y_offset_px
        return self
    
    def read(self):
        return self._idx_select, self._titles[self._idx_select]
    
    def _make_base_image(self, bar_width):
        
        # For convenience
        num_titles = len(self._titles)
        btn_width = bar_width / num_titles
        btn_height = self.height_px
        btn_font = cv2.FONT_HERSHEY_SIMPLEX
        font_thick = 1
        line_type = cv2.LINE_AA
        
        # Make new (blank) image to draw into
        base_img = np.full((self.height_px, bar_width, 3), self._bg_color, dtype = np.uint8)
        
        title_font_lut = {}
        for idx, title in enumerate(self._titles):
            
            # Find text sizing that fits inside button
            for font_scale in [1, 0.8, 0.5, 0.35, 0.1]:
                (txt_w, txt_h), txt_baseline = cv2.getTextSize(title, btn_font, font_scale, font_thick)
                is_too_big = (txt_w > 0.8*btn_width) or (txt_h > 0.8 * btn_height)
                if not is_too_big:
                    break
            
            # Figure out text positioning
            x1 = idx * btn_width
            x2 = (idx+1) * btn_width
            bounds_tl = (int(x1), int(-5))
            bounds_br = (int(x2), int(btn_height + 5))
            mid_x = int(round((x1 + x2) * 0.5))
            mid_y = btn_height // 2
            txt_x = int(mid_x - txt_w//2)
            txt_y = int(mid_y + txt_baseline)
            txt_xy = (txt_x, txt_y)
            
            # Record all text drawing config, so we can re-use for selections
            font_config = {
                "fontFace": btn_font,
                "fontScale": font_scale,
                "text": title,
                "org": txt_xy,
                "lineType": line_type,
            }
            title_font_lut[title] = font_config
            
            # Draw text with bounding box
            cv2.putText(base_img, **font_config, color = self._fg_color, thickness=1)
            cv2.rectangle(base_img, bounds_tl, bounds_br, self._line_color)
        
        # Record text config, so we can draw selection highlights later!
        self._title_font_lut = title_font_lut
        
        return base_img
    
    def draw_selection(self, selected_title):
        
        display_copy = self._base_img.copy()
        selected_font_config = self._title_font_lut.get(selected_title, None)
        if selected_font_config is not None:
            cv2.putText(display_copy, **selected_font_config, color=self._select_color, thickness=2)
        
        return display_copy
    
    def prepend_to_frame(self, display_frame):
        
        disp_w, base_w = display_frame.shape[1], self._base_img.shape[1]
        size_mismatch = disp_w != base_w
        if size_mismatch:
            self._base_img = self._make_base_image(disp_w)
        
        selected_title = self._titles[self._idx_select]
        select_img = self.draw_selection(selected_title)
        
        select_h = select_img.shape[0]
        self._interact_y1y2 = (0, select_h)
        
        return np.vstack((select_img, display_frame))
    
    def append_to_frame(self, display_frame):
        
        disp_h, disp_w = display_frame.shape[0:2]
        base_w = self._base_img.shape[1]
        size_mismatch = disp_w != base_w
        if size_mismatch:
            self._base_img = self._make_base_image(disp_w)
        
        selected_title = self._titles[self._idx_select]
        select_img = self.draw_selection(selected_title)
        
        select_h = select_img.shape[0]
        self._interact_y1y2 = (disp_h, disp_h + select_h)
        
        return np.vstack((display_frame, select_img))


# ---------------------------------------------------------------------------------------------------------------------
#%% Set up video source

video_source = None
if arg_use_webcam:
    video_source = 0
    
elif arg_rtsp_url is not None:
    video_source = arg_rtsp_url
    
else:
    # Ask user for rtsp url, if no other input was selected
    print("")
    video_source = input("Enter rtsp url: ").strip()


# ---------------------------------------------------------------------------------------------------------------------
#%% Set up models

aruco_model = ArucoDemo()
depth_model = DepthDemo("models/depth")
pose_model = PoseDemo("models/pose")


# ---------------------------------------------------------------------------------------------------------------------
#%% Video Loop

# Set up camera access
vread = VideoReader(video_source)
vread.exhaust_buffered_frames()

# Set up display scaling, if needed
needs_scaling = arg_display_size is not None
if needs_scaling:
    video_h, video_w = vread.shape[0:2]
    max_video_size = max(video_h, video_w)
    scale_factor = arg_display_size / max_video_size
    scaled_w = int(round(scale_factor * video_w))
    scaled_h = int(round(scale_factor * video_h))
    scaled_wh = (scaled_w, scaled_h)

# Create all selection bars (model select + variant selectors for each model)
variant_select_bg_color = (30,30,30)
select_bar = SelectionBar("Pose", "ArUco", "Depth", "Pose + ArUco", "All", select_color=(0,120,255))
pose_select_bar = SelectionBar("Fastest", "Small", "Medium", bg_color=variant_select_bg_color)
aruco_select_bar = SelectionBar("4x4 (250)", "5x5 (250)", "6x6 (250)", "7x7 (250)", bg_color=variant_select_bg_color)
depth_select_bar = SelectionBar("Small", "Medium", "Medium (Hires)", bg_color=variant_select_bg_color)

# Set up bar collection for managing enable/disable control
bar_lut = {"Pose": pose_select_bar, "ArUco": aruco_select_bar, "Depth": depth_select_bar}
for bar_ref in bar_lut.values():
    bar_ref.enable(False)
    bar_ref.set_y_offset(select_bar.height_px)
prev_select = None

# Create window & attach selection bar callbacks
window = DisplayWindow("Pacefactory")
window.add_callbacks(select_bar, aruco_select_bar, pose_select_bar, depth_select_bar)

try:
    for frame in vread:
        
        if needs_scaling:
            frame = cv2.resize(frame, dsize=scaled_wh)
        
        _, model_select = select_bar.read()
        match model_select:
            
            case "Pose":
                idx_select, _ = pose_select_bar.read()
                pose_model.set_model_select(idx_select)
                
                pose_results = pose_model.process_frame(frame)
                frame = pose_model.draw_results(pose_results, frame)
                frame = pose_select_bar.append_to_frame(frame)
            
            case "ArUco":
                idx_select, _ = aruco_select_bar.read()
                aruco_model.set_model_select(idx_select)
                
                aru_results = aruco_model.process_frame(frame)
                frame = aruco_model.draw_results(aru_results, frame)
                frame = aruco_select_bar.append_to_frame(frame)
            
            case "Depth":
                idx_select, _ = depth_select_bar.read()
                depth_model.set_model_select(idx_select)
                
                depth_result = depth_model.process_frame(frame)
                frame = depth_model.draw_results(depth_result, frame.shape)
                frame = depth_select_bar.append_to_frame(frame)
            
            case "Pose + ArUco":
                aru_results = aruco_model.process_frame(frame)
                pose_results = pose_model.process_frame(frame)
                frame = aruco_model.draw_results(aru_results, frame)
                frame = pose_model.draw_results(pose_results, frame)
            
            case "All":
                depth_result = depth_model.process_frame(frame)
                aru_results = aruco_model.process_frame(frame)
                pose_results = pose_model.process_frame(frame)
                frame = depth_model.draw_results(depth_result, frame.shape)
                frame = aruco_model.draw_results(aru_results, frame)
                frame = pose_model.draw_results(pose_results, frame)
            
            case _:
                pass
        
        # Display image with model selection bar header
        display_frame = select_bar.prepend_to_frame(frame)
        req_close, keypress = window.imshow(display_frame)
        if req_close:
            break
        
        # Hacky-ish code to disable all but the currently selected model menu bar
        selection_changed = model_select != prev_select
        if selection_changed:
            prev_select = model_select
            for bar_ref in bar_lut.values():
                bar_ref.enable(False)
            bar_to_enable = bar_lut.get(model_select, None)
            if bar_to_enable is not None:
                bar_to_enable.enable(True)
        
        pass

except KeyboardInterrupt:
    print("Cancelled by Ctrl+C")

except Exception as err:
    cv2.destroyAllWindows()
    vread.release()
    raise err

# Clean up
cv2.destroyAllWindows()
vread.release()
