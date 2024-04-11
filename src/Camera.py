import dearpygui.dearpygui as dpg
import math
import numpy as np
import trimesh
import os
import json


class Camera:

    def __init__(self, pos=dpg.mvVec4(0.0, 0.0, 30.0, 1.0), pitch=0.0, yaw=0.0):
        self.pos = pos
        self.pitch = pitch
        self.yaw = yaw
        self.moving = False
        self.mouse_pos = [0, 0]
        self.dirty = False
        self.field_of_view = 45.0
        self.nearClip = 0.01
        self.farClip = 400.0

    def toggle_moving(self):
        self.moving = not self.moving

    def mark_dirty(self):
        self.dirty = True

    def view_matrix(self):
        if type(self.pos) != "list":
            new_pos = [self.pos[0], self.pos[1], self.pos[2], self.pos[3]]
        return dpg.create_fps_matrix(new_pos, self.pitch, self.yaw)

    def projection_matrix(self, width, height):
        return dpg.create_perspective_matrix(math.pi * self.field_of_view / 180.0, width / height, self.nearClip,
                                             self.farClip)

    def _set_field_of_view(self, value):
        self.field_of_view = float(value)
        self.dirty = True

    def _set_near(self, value):
        self.nearClip = value
        self.dirty = True

    def _set_far(self, value):
        self.farClip = value
        self.dirty = True

    # def show_controls(self):
    #     with dpg.window(label="Camera Controls", width=500, height=500):
    #         dpg.add_text(str(self.pos[0]), label="Camera X", show_label=True, tag="Camera X")
    #         dpg.add_text(str(self.pos[1]), label="Camera Y", show_label=True, tag="Camera Y")
    #         dpg.add_text(str(self.pos[2]), label="Camera Z", show_label=True, tag="Camera Z")
    #         dpg.add_text("0", label="Camera Yaw", show_label=True, tag="Camera Yaw")
    #         dpg.add_text("0", label="Camera Pitch", show_label=True, tag="Camera Pitch")
    #         dpg.add_slider_float(label="near_clip", min_value=0.1, max_value=100, default_value=self.nearClip,
    #                              callback=lambda s, a: self._set_near(a))
    #         dpg.add_slider_float(label="far_clip", min_value=0.2, max_value=1000, default_value=self.farClip,
    #                              callback=lambda s, a: self._set_far(a))
    #         dpg.add_text("field_of_view")
    #         dpg.add_radio_button(["45.0", "60.0", "90.0"], default_value="45.0", label="field of view",
    #                              callback=lambda s, a: self._set_field_of_view(a))
