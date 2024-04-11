import argparse
import os

import dearpygui.dearpygui as dpg
from screeninfo import get_monitors

from Configs.config import Config
from src.Annotator import Annotator
from src.Camera import Camera
from src.PointCloud import PointCloud
from src.utils import *


class Main:
    def __init__(self, config):
        self.config = config
        self.reload = False
        self.file_name = self.config.initialize.file_name
        self.file_folder = self.config.initialize.file_folder
        self.scene_id = self.config.initialize.scene_id
        self.annotator = None
        if self.config.label_type == "raw":
            self.object_names = read_text_class(os.path.join(self.config.scannet_path, "object_fixed_label.txt"))
        else:
            self.object_names = read_text_class(os.path.join(self.config.scannet_path, "object_label.txt"))
        self.relation_names = read_text_class(os.path.join(self.config.scannet_path, "relation_label.txt"))
        self.width = dpg.get_viewport_client_width()
        self.height = dpg.get_viewport_client_height()
        self.camera = Camera(dpg.mvVec4(0.0, 0.0, 30.0, 1.0), 0.0, 0.0)

    def _reload_func(self, sender, value):
        print(sender, value)
        self.reload = True
        self.file_name = value["file_name"]
        self.file_folder = value["current_path"]
        self.scene_id = value["file_name"][:12]
        points, colors, instances, labels, face = load_mesh(os.path.join(self.file_folder, self.file_name),
                                                            self.config.label_type)
        instances_to_color, instances_to_label = get_instance_color_label_dict(instances, colors, labels)

        self.annotator.point_cloud.initialize(points, face, colors, instances, labels, instances_to_color,
                                              instances_to_label)
        self.annotator.initialize(instances, instances_to_label, self.file_folder)

    def reset_view_matrix(self):
        self.annotator.point_cloud._set_rotation([0.0, 0.0, 0.0])
        self.annotator.point_cloud._set_position([0.0, 0.0, 0.0])
        self.annotator.point_cloud._set_scale([1.0, 1.0, 1.0])
        projection = self.camera.projection_matrix(self.width / 2, self.height * 7 / 8)
        self.annotator.point_cloud.update(projection, self.view)

    def set_main_window(self, rect):
        with dpg.window(tag="Primary Window"):
            with dpg.file_dialog(directory_selector=False, show=False, callback=self._reload_func, id="file_dialog_id",
                                 width=int(self.config.main_screen[0] * 0.5),
                                 height=int(self.config.main_screen[1] * 0.5),
                                 file_count=1, modal=True, default_path=self.file_folder):
                dpg.add_file_extension("", color=(150, 255, 150, 255))
                dpg.add_file_extension(".ply", color=(255, 100, 100, 255), custom_text="[Ply]")
                # dpg.add_file_extension(".py", color=(0, 255, 0, 255), custom_text="[Python]")

            with dpg.menu_bar():
                with dpg.menu(label="Files"):
                    dpg.add_menu_item(label="save result", callback=self.annotator.save_annotation_result)
                    dpg.add_menu_item(label="files", callback=lambda: dpg.show_item("file_dialog_id"))
                with dpg.menu(label="View"):
                    dpg.add_menu_item(label="reset view", callback=self.reset_view_matrix)

            with dpg.drawlist(width=self.width, height=self.height, tag="pc_drawlist"):
                self.annotator.point_cloud.submit()  # 在展示图层上画点云

                # for c in cubes:
                #     c.submit(cube.layer)

                dpg.draw_rectangle((0, 0), (10, 10), tag=rect)  # 画矩形，画线
                with dpg.draw_layer(perspective_divide=True, tag="gizmo_layer"):
                    with dpg.draw_node(tag="gizmo"):
                        dpg.draw_line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), color=(255, 0, 0))
                        dpg.draw_line((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), color=(0, 255, 0))
                        dpg.draw_line((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), color=(0, 0, 255))

        dpg.set_primary_window("Primary Window", True)

    def load_pointcloud_annotator(self):
        points, colors, instances, labels, face = load_mesh(os.path.join(self.file_folder, self.file_name),
                                                            self.config.label_type)
        instances_to_color, instances_to_label = get_instance_color_label_dict(instances, colors, labels)
        self.annotator = Annotator(instances, self.relation_names, self.object_names, instances_to_label, self.scene_id,
                                   self.file_folder, self.config)
        self.annotator.point_cloud = PointCloud(self.object_names, points, colors, [0, 0], instances, labels,
                                                instances_to_color, instances_to_label, face, "Primary Window",
                                                "pc_drawlist")

        self.annotator.show_controls()
        self.annotator.point_cloud.show_controls()

    def main_loop(self):
        # self.camera.show_controls()

        self.load_pointcloud_annotator()

        dpg.set_viewport_resize_callback(self.camera.mark_dirty)
        rect = dpg.generate_uuid()

        self.set_main_window(rect)  # 设置主窗口下的items

        with dpg.handler_registry(tag="__demo_keyboard_handler"):
            # 轮询，判断键盘的输入
            dpg.add_key_down_handler(key=dpg.mvKey_W,
                                     callback=lambda s, a, u: self.annotator.point_cloud.update_pos(s, a, u))
            dpg.add_key_down_handler(key=dpg.mvKey_S,
                                     callback=lambda s, a, u: self.annotator.point_cloud.update_pos(s, a, u))
            dpg.add_key_down_handler(key=dpg.mvKey_D,
                                     callback=lambda s, a, u: self.annotator.point_cloud.update_pos(s, a, u))
            dpg.add_key_down_handler(key=dpg.mvKey_A,
                                     callback=lambda s, a, u: self.annotator.point_cloud.update_pos(s, a, u))

            # 这三行的逻辑就是当我按住鼠标右键的时候就可以拖动3D场景
            dpg.add_mouse_move_handler(callback=lambda s, a, u: self.annotator.point_cloud.move_handler(s, a, u))
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Right,
                                        callback=lambda: self.annotator.point_cloud.toggle_moving())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Right,
                                          callback=lambda: self.annotator.point_cloud.toggle_moving())
            dpg.add_mouse_wheel_handler(callback=lambda s, a: self.annotator.point_cloud.update_scale_with_mouse(s, a))

        # main loop
        dpg.show_viewport()
        dpg.set_exit_callback(self.annotator.save_annotation_result)
        self.view = self.camera.view_matrix()
        while dpg.is_dearpygui_running():

            if self.camera.dirty or self.annotator.point_cloud.dirty:
                self.width = dpg.get_viewport_client_width()
                self.height = dpg.get_viewport_client_height()
                self.annotator.point_cloud.update_clip_space(self.width / 18, self.height / 18, self.width / 2,
                                                             self.height * 15 / 18,
                                                             -1.0, 1.0)  # 左上角坐标，宽度 长度
                dpg.configure_item(rect, pmin=(self.width / 18, self.height / 18),
                                   pmax=(self.width * 5 / 9, self.height * 8 / 9))  # 重新定义矩形框大小，左上角和右下角

                projection = self.camera.projection_matrix(self.width / 2, self.height * 15 / 18)  # 区域的宽和长

                dpg.apply_transform("gizmo", projection * self.view)
                self.annotator.point_cloud.update_center([self.width * 11 / 36, self.height * 17 / 36], self.width / 2,
                                                         self.height * 15 / 18)  # 中心点坐标，宽，长
                # print(projection * view)
                dpg.set_clip_space("gizmo_layer", self.width / 18, self.height / 18, self.width / 2,
                                   self.height * 15 / 18, -1.0, 1.0)  # 左上角宽 高

                dpg.configure_item("pc_drawlist", width=self.width, height=self.height * 25 / 26)

                self.annotator.point_cloud.update(projection, self.view)
                # for c in cubes:
                #     c.update(projection, view)

                self.camera.dirty = False
                self.annotator.point_cloud.dirty = False

            dpg.render_dearpygui_frame()


def load_config():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='Configs/config.json',
                        help='configuration file name. Relative path under given path (default: config.yml)')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = Config(config_path)

    monitors = get_monitors()
    config.main_screen = [int(monitors[0].width * 0.8), int(monitors[0].height * 0.8)]
    config.root = os.path.dirname(__file__)

    return config


if __name__ == "__main__":
    config = load_config()

    dpg.create_context()
    with dpg.font_registry():
        font = dpg.add_font("Configs/Roboto-Regular-14.ttf", config.font_size)
        dpg.bind_font(font)

    dpg.configure_app(init_file="Configs/custom_layout.ini")
    # dpg.configure_app()
    dpg.create_viewport(title='Custom Window Size', width=config.main_screen[0],
                        height=config.main_screen[1])
    dpg.setup_dearpygui()

    main = Main(config)
    main.main_loop()

    dpg.destroy_context()
