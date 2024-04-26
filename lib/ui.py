#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class SelectionBar:
    
    def __init__(self, *titles, select_color = (0,255,0), bar_height = 60,
                 fg_color = (255,255,255), bg_color = (40,40,40), line_color = (80,80,80)):
        
        # Storage for selection entries
        self._titles = list(titles)
        
        # Graphics
        self.height_px = bar_height
        self._fg_color = fg_color
        self._bg_color = bg_color
        self._select_color = select_color
        self._line_color = line_color
        self._base_img = np.full((1,1,3), 0, dtype = np.uint8)
        
        # Interaction settings
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
    
    def read_index(self):
        return self._idx_select
    
    def read(self):
        return self._titles[self._idx_select]
    
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
