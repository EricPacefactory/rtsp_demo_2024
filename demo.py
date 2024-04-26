#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import argparse

import cv2
import numpy as np

from lib.display import DisplayWindow
from lib.video import PlaybackBar, make_video_reader
from lib.ui import SelectionBar
from lib.misc import SourceHistory

from lib.aruco_demo_wrapper import ArucoDemo
from lib.pose_demo_wrapper import PoseDemo
from lib.depth_demo_wrapper import DepthDemo


# ---------------------------------------------------------------------------------------------------------------------
#%% Script args

# Set script arg defaults (can change these for easier debugging/development work!)
default_video_source = None
default_display_size_px = 1000

# Define script arguments
parser = argparse.ArgumentParser(description="Demo script for running pose/ArUco/depth models on live video")
parser.add_argument("-i", "--video_source", default=default_video_source, type=str,
                    help="Video source (rtsp url, video file, image file or 0 for webcam")
parser.add_argument("-s", "--display_size", default=default_display_size_px, type=int,
                    help=f"Set maximum side length for displayed image (default: {default_display_size_px})")
    
# For convenience
args = parser.parse_args()
arg_video_source = args.video_source
arg_display_size = args.display_size

# Set up video source history loading/saving
history = SourceHistory()
prev_source = history.load()

# ---------------------------------------------------------------------------------------------------------------------
#%% Set up models

pose_model = PoseDemo()
aruco_model = ArucoDemo()
depth_model = DepthDemo()


# ---------------------------------------------------------------------------------------------------------------------
#%% Set up video source

# Ask user for rtsp url, if no other input was selected
video_source = arg_video_source
if video_source is None:
    print("",
          "This script uses yolo models from ultralytics",
          "Other models can be downloaded from:",
          "https://github.com/ultralytics/ultralytics?tab=readme-ov-file#models",
          "",
          "Depth models are based on onnx versions of 'depth-anything'",
          "Models can be downloaded from:",
          "https://github.com/fabio-sim/Depth-Anything-ONNX/releases",
          "",
          "Please enter a video source",
          "- For webcam use, enter 0 (or 1, 2, etc. if you have multiple webcams)",
          "- For a video file, enter the path to the file",
          "- You can also enter a path to an image",
          "- Or enter an rtsp url, eg. rtsp://user:password@192.168.0.100:554/profile1",
          "", "", sep = "\n", flush=True)
    
    # Provide default (if present)
    have_default = (prev_source is not None)
    if have_default: print(f"   (default): {prev_source}")
    video_source = input("Video source: ").strip()
    if video_source == "" and have_default: video_source = prev_source


# ---------------------------------------------------------------------------------------------------------------------
#%% Video Loop

# Keycodes, for clarity
KEY_UPARROW = 82
KEY_DOWNARROW = 84

# Set up frame reading
source_type, vread = make_video_reader(video_source)
history.save(video_source)

# Set up playback control, if needed
playback_bar = PlaybackBar(vread)
playback_bar.enable(source_type == "video")

# Set up display scaling
video_h, video_w, _ = vread.get_shape()
scaled_display_size = arg_display_size
max_video_size = max(video_h, video_w)
scale_factor = scaled_display_size / max_video_size

# Create model selection bar
header_select_bar = SelectionBar("Pose", "ArUco", "Depth", "Pose + ArUco", "All", select_color=(0,120,255))

# Create model-specific variant selection bars
VariantBar = lambda *button_labels: SelectionBar(*button_labels, bg_color=(30,30,30))
pose_select_bar = VariantBar(*pose_model.get_model_names())
aruco_select_bar = VariantBar(*aruco_model.get_model_names())
depth_select_bar = VariantBar(*depth_model.get_model_names())

# Set up bar collection for managing enable/disable control
bar_lut = {"Pose": pose_select_bar, "ArUco": aruco_select_bar, "Depth": depth_select_bar}
for bar_ref in bar_lut.values():
    bar_ref.enable(False)
    bar_ref.set_y_offset(header_select_bar.height_px)
prev_select = None

# Create window & attach selection bar callbacks
window = DisplayWindow("Pacefactory - q to quit")
window.add_callbacks(header_select_bar, aruco_select_bar, pose_select_bar, depth_select_bar, playback_bar)

# Some feedback
print("",
      "Displaying video!",
      "  - Press up/down arrow keys to resize the display",
      "  - Press esc or q to quit",
      sep = "\n", flush=True)
try:
    for frame in vread:
        
        frame = cv2.resize(frame, dsize=None, fx=scale_factor, fy=scale_factor)
        
        model_select = header_select_bar.read()
        match model_select:
            
            case "Pose":
                pose_select = pose_select_bar.read()
                pose_model.set_model_select(pose_select)
                
                pose_results = pose_model.process_frame(frame)
                frame = pose_model.draw_results(pose_results, frame)
                frame = pose_select_bar.append_to_frame(frame)
            
            case "ArUco":
                aru_select = aruco_select_bar.read()
                aruco_model.set_model_select(aru_select)
                
                aru_results = aruco_model.process_frame(frame)
                frame = aruco_model.draw_results(aru_results, frame)
                frame = aruco_select_bar.append_to_frame(frame)
            
            case "Depth":
                depth_select = depth_select_bar.read()
                depth_model.set_model_select(depth_select)
                
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
                print("UNKNOWN MODEL SELECTION:", model_select)
        
        # Display image with model selection bar header
        display_frame = header_select_bar.prepend_to_frame(frame)
        display_frame = playback_bar.append_to_frame(display_frame)
        req_close, keypress = window.imshow(display_frame)
        if req_close:
            break
        
        # Change display size on keypress
        if keypress == KEY_UPARROW:
            scaled_display_size = min(4000, scaled_display_size + 50)
            scale_factor = scaled_display_size / max_video_size
        if keypress == KEY_DOWNARROW:
            scaled_display_size = max(100, scaled_display_size - 50)
            scale_factor = scaled_display_size / max_video_size
        
        # Hacky-ish code to disable all but the currently selected model menu bar
        # -> without this step, selection buttons will respond to clicks, even if not rendered!
        # -> e.g. changing the aruco marker size can also change the yolo pose model
        selection_changed = model_select != prev_select
        if selection_changed:
            prev_select = model_select
            for bar_ref in bar_lut.values():
                bar_ref.enable(False)
            bar_to_enable = bar_lut.get(model_select, None)
            if bar_to_enable is not None:
                bar_to_enable.enable(True)
        
        # Control playback of video files
        playback_bar.adjust_playback_on_drag()

except KeyboardInterrupt:
    print("Cancelled by Ctrl+C")

finally:
    # Clean up
    vread.release()
    cv2.destroyAllWindows()
