import math

import dearpygui.dearpygui as dpg

from src.utils import *


class PointCloud:

    def __init__(self, object_names, points, colors, center, instances, labels,
                 instances_to_color, instances_to_label, face, primary_window, pc_drawlist, config, pos=[0, 0, 0],
                 radis=1.2):
        self.config = config

        self.points, self.colors, self.instances, self.face, self.primary_window, self.pc_drawlist = \
            points, colors, instances, face, primary_window, pc_drawlist
        self.instances_names = np.unique(instances)  # 场景中有哪些instances
        self.distance_dict = get_distance_dict(points, instances, self.instances_names, instances_to_label,
                                               object_names, radis)
        self.instances_node_list = []
        self.object_names = object_names
        self.labels = labels
        self.instances_to_color, self.instances_to_label = instances_to_color, instances_to_label
        self._initial_windows()

        self.outline_color = [255, 255, 255, 0]

        self.depth_clipping = False
        self.perspective_divide = True
        self.cull_mode = dpg.mvCullMode_None
        self.dirty = False  # 应该是如果视角或者任何东西发生了改变，dirty就会true，然后就会在while循环中重现展示物体
        self.pos = pos
        self.rot = [0, 0, 0]
        self.scale = [1.0, 1.0, 1.0]
        self.layer = dpg.generate_uuid()
        self.node = dpg.generate_uuid()
        self.center = center
        self.moving = False
        self.mouse_pos = [0, 0]
        self.rotation_speed = 0.002
        self.radis = radis
        self.travel_speed = 0.1
        self.window_width = 0.0
        self.window_height = 0.0
        self.show_mode = "point_cloud"  # 物体的展示格式：point_cloud和mesh
        self.last_highlight_sub = None
        self.last_highlight_obj = None  # 记录上一个高亮的Instance

    def initialize(self, points, face, colors, instances, labels, instances_to_color,
                   instances_to_label):
        self.points = points
        self.face = face
        self.colors = colors
        self.instances = instances
        self.instances_to_color, self.instances_to_label = instances_to_color, instances_to_label
        self.instances_names = np.unique(instances)
        self.distance_dict = get_distance_dict(points, instances, self.instances_names, self.instances_to_label,
                                               self.object_names, self.radis)
        self.configer_items()

    def update_center(self, center, width, height):
        self.center = center
        self.window_height = height
        self.window_width = width

    def update_pos(self, sender, key, user):
        if key[0] == dpg.mvKey_D:
            self.pos[0] = self.pos[0] - self.travel_speed * 2
        if key[0] == dpg.mvKey_A:
            self.pos[0] = self.pos[0] + self.travel_speed * 2
        if key[0] == dpg.mvKey_W:
            self.pos[1] = self.pos[1] - self.travel_speed * 2
        if key[0] == dpg.mvKey_S:
            self.pos[1] = self.pos[1] + self.travel_speed * 2

        self.dirty = True

    def update_scale_with_mouse(self, sender, key):
        mouse_pose = dpg.get_mouse_pos(local=False)
        # print(mouse_pose)
        # print(self.window_height)
        # print(self.window_width)
        if mouse_pose[0] < self.window_width * 9 / 8 and mouse_pose[1] < self.window_height * 9 / 8:
            self.scale[0] = self.scale[0] + key * self.travel_speed
            self.scale[1] = self.scale[1] + key * self.travel_speed
            self.scale[2] = self.scale[2] + key * self.travel_speed

            self.dirty = True

    def submit(self, layer=None):

        if layer is None:
            dpg.push_container_stack(  # 加入到容器栈当中
                dpg.add_draw_layer(tag=self.layer, depth_clipping=True, cull_mode=dpg.mvCullMode_Back,
                                   perspective_divide=True))
            self._reconfigure()
        else:
            dpg.push_container_stack(layer)

        with dpg.draw_node(tag=self.node):
            self.instances_node_list = [dpg.add_draw_node(tag=f"{i}_ins") for i in self.instances_names]
            dpg.add_draw_node(tag="temp_sub_ins")
            dpg.add_draw_node(tag="temp_obj_ins")

        # 两种展示格式
        if self.show_mode == "point_cloud":
            Size = min(self.points.shape[0] * 0.7, self.config.max_show_points)
            t_indices = np.random.choice(self.points.shape[0], size=Size, replace=True)
            t_points = self.points[t_indices, :]
            t_colors = self.colors[t_indices, :].tolist()
            t_instances = self.instances[t_indices]
            for i, point in enumerate(t_points):
                dpg.draw_circle((point[0], point[1], point[2]),
                                self.config.pc_radius, color=t_colors[i], fill=t_colors[i], show=True,
                                parent=f"{t_instances[i]}_ins")

        if self.show_mode == "mesh":
            t_colors = self.colors.tolist()
            for i, triangle in enumerate(self.face):
                dpg.draw_triangle(
                    (self.points[triangle[0]][0], self.points[triangle[0]][1], self.points[triangle[0]][2]),
                    (self.points[triangle[1]][0], self.points[triangle[1]][1], self.points[triangle[1]][2]),
                    (self.points[triangle[2]][0], self.points[triangle[2]][1], self.points[triangle[2]][2]),
                    color=t_colors[triangle[0]], fill=t_colors[triangle[0]], show=True, thickness=1,
                    parent=f"{self.instances[triangle[0]]}_ins")

        # # draw_bbox
        # for ins in self.instances_names:
        #     index = np.where(self.instances == ins)
        #     t_p = self.points[index]
        #     point_list = get_bbox(t_p)
        #     for i in range(1, 9):
        #         for j in range(i, 9):
        #             if np.sum(point_list[i] != point_list[j]) >= 2:
        #                 continue
        #             dpg.draw_line(point_list[i], point_list[j], color=[0, 255, 0, 100], show=True, parent=f"{ins}_ins")

        dpg.pop_container_stack()

    def configer_items(self):
        """读入一个新的点云后对点云相关界面进行修改"""
        dpg.delete_item(self.layer, children_only=False)
        dpg.push_container_stack(item=self.primary_window)
        dpg.push_container_stack(item=self.pc_drawlist)
        self.submit()
        dpg.pop_container_stack()
        dpg.pop_container_stack()

        dpg.delete_item("window_1", children_only=True)
        dpg.delete_item("window_0", children_only=True)
        self.show_controls()

    def draw(self, indices, instance, s_or_o):
        if self.show_mode == "point_cloud":
            for i, point in enumerate(self.points[indices]):
                dpg.draw_circle((point[0], point[1], point[2]),
                                4, color=[255, 0, 0, 255], fill=[255, 0, 0, 255], show=True,
                                parent=f"temp_{s_or_o}_ins")

        if self.show_mode == "mesh":
            for i, triangle in enumerate(self.face):
                if triangle[0].item() in indices:
                    dpg.draw_triangle(
                        (self.points[triangle[0]][0], self.points[triangle[0]][1], self.points[triangle[0]][2]),
                        (self.points[triangle[1]][0], self.points[triangle[1]][1], self.points[triangle[1]][2]),
                        (self.points[triangle[2]][0], self.points[triangle[2]][1], self.points[triangle[2]][2]),
                        color=[255, 0, 0, 255], fill=[255, 0, 0, 255], show=True, thickness=1,
                        parent=f"temp_{s_or_o}_ins")

    def highlight(self, instance, s_or_o):
        if s_or_o == "sub":
            dpg.delete_item(item="temp_sub_ins", children_only=True)
            if self.last_highlight_sub is not None:
                dpg.configure_item(item=f"{self.last_highlight_sub}_ins", show=True)
            self.last_highlight_sub = instance  # 更新Instance
            dpg.configure_item(item=f"{instance}_ins", show=False)  # 现将当前的Instance隐藏
            indices = np.where(self.instances == instance)[0]
            self.draw(indices, instance, s_or_o)

        if s_or_o == "obj":
            dpg.delete_item(item="temp_obj_ins", children_only=True)
            if self.last_highlight_obj is not None:
                dpg.configure_item(item=f"{self.last_highlight_obj}_ins", show=True)
            self.last_highlight_obj = instance  # 更新Instance
            dpg.configure_item(item=f"{instance}_ins", show=False)  # 现将当前的Instance隐藏
            indices = np.where(self.instances == instance)[0]
            self.draw(indices, instance, s_or_o)

    def _set_seen_instance(self, sender, value):
        # print(sender)
        # print(f"{sender}_ins")
        # print(self.instances_node_list)
        # print(np.where(self.instances_names == int(sender)))
        # print(self.instances_node_list[0])
        dpg.configure_item(item=f"{sender}_ins", show=value)

    def _set_show_mode(self, sender, value):
        self.show_mode = value
        dpg.delete_item(self.layer, children_only=False)
        dpg.push_container_stack(item=self.primary_window)
        dpg.push_container_stack(item=self.pc_drawlist)
        self.submit()
        dpg.pop_container_stack()
        dpg.pop_container_stack()
        # print(value)

    def toggle_moving(self):
        self.moving = not self.moving

    def move_handler(self, sender, pos, user):
        if self.moving:
            dx = self.mouse_pos[0] - pos[0]
            dy = self.mouse_pos[1] - pos[1]

            if (abs(pos[0] - self.center[0]) < (self.window_width / 2) and
                    abs(pos[1] - self.center[1]) < (self.window_height / 2)):
                place = "in"  # 在点云范围内
            elif (abs(pos[0] - self.center[0]) < (self.window_width / 2) and
                  abs(pos[1] - self.center[1]) > (self.window_height / 2)):
                place = "down"
            else:
                place = "out"

            if dx != 0.0 or dy != 0.0:
                # print(dx, dy, place)
                self.rotate(dx, dy, place)

        self.mouse_pos = pos

    def rotate(self, dx, dy, place):
        if place == "in":
            self.rot[0] = self.rot[0] + dy * math.pi * self.rotation_speed * 0.7
            self.rot[1] = self.rot[1] + dx * math.pi * self.rotation_speed
        elif place == "down":
            self.rot[2] = self.rot[2] - dx * math.pi * self.rotation_speed
            self.rot[0] = self.rot[0] + dy * math.pi * self.rotation_speed * 0.7
        else:
            self.rot[1] = self.rot[1] + dx * math.pi * self.rotation_speed
            self.rot[2] = self.rot[2] + dy * math.pi * self.rotation_speed * 0.7

        self.dirty = True

    def _reconfigure(self):
        dpg.configure_item(self.layer, depth_clipping=self.depth_clipping, perspective_divide=self.perspective_divide,
                           cull_mode=self.cull_mode)

    def _set_depth_clipping(self, value):
        self.depth_clipping = value
        self._reconfigure()

    def _set_perspective_divide(self, value):
        self.perspective_divide = value
        self._reconfigure()

    def _set_cull_mode(self, value):

        if value == "mvCullMode_None":
            self.cull_mode = dpg.mvCullMode_None
        elif value == "mvCullMode_Front":
            self.cull_mode = dpg.mvCullMode_Front
        elif value == "mvCullMode_Back":
            self.cull_mode = dpg.mvCullMode_Back
        self._reconfigure()

    def _set_rotation(self, value):
        self.rot = value
        self.dirty = True

    def _set_position(self, value):
        self.pos = value
        self.dirty = True

    def _set_scale(self, value):
        self.scale = value
        self.dirty = True

    def update_clip_space(self, top_left_x, top_left_y, width, height, min_depth, max_depth):
        dpg.set_clip_space(self.layer, top_left_x, top_left_y, width, height, min_depth, max_depth)

    def _initial_windows(self):
        self.window_0 = dpg.add_window(label="Controls", width=500, height=500, tag="window_0")
        self.window_1 = dpg.add_window(label="Instance_list", width=500, height=500, tag="window_1")

    def show_controls(self):
        dpg.push_container_stack(self.window_0)
        dpg.add_checkbox(label="depth_clipping", default_value=self.depth_clipping,
                         callback=lambda s, a: self._set_depth_clipping(a))
        # dpg.add_checkbox(label="perspective_divide", default_value=self.perspective_divide,
        #                  callback=lambda s, a: self._set_perspective_divide(a))
        dpg.add_text("cull_mode")
        # dpg.add_radio_button(["mvCullMode_None", "mvCullMode_Back", "mvCullMode_Front"],
        #                      default_value="mvCullMode_None", label="cull_mode",
        #                      callback=lambda s, a: self._set_cull_mode(a))
        dpg.add_text("show_mode")
        dpg.add_radio_button(["mesh", "point_cloud"],
                             default_value="point_cloud", label="show_mode",
                             callback=self._set_show_mode)
        dpg.add_slider_floatx(label="Position", size=3, callback=lambda s, a: self._set_position(a))
        dpg.add_slider_floatx(label="Rotation", size=3, min_value=-math.pi, max_value=math.pi,
                              callback=lambda s, a: self._set_rotation(a))
        dpg.add_slider_floatx(label="Scale", size=3, max_value=10.0, default_value=self.scale,
                              callback=lambda s, a: self._set_scale(a))
        dpg.pop_container_stack()

        dpg.push_container_stack(self.window_1)
        for i in self.instances_names:
            with dpg.group(horizontal=True):
                checkbox_id = dpg.add_checkbox(tag=f"{i}",
                                               label=f"{i}-{self.object_names[self.instances_to_label[i] - 1]}",
                                               default_value=True,
                                               callback=lambda s, a: self._set_seen_instance(s, a))

                with dpg.drawlist(width=100, height=20):
                    checkbox_rect = dpg.get_item_rect_size(checkbox_id)  # 获取复选框的位置和大小
                    drawlist_pos = dpg.get_item_rect_min(checkbox_id)  # 获取 Drawlist 的位置
                    rect_pos = (drawlist_pos[0] + checkbox_rect[0] + 0, drawlist_pos[1])  # 计算矩形的位置（在复选框右侧）
                    dpg.draw_rectangle(pmin=rect_pos, pmax=(rect_pos[0] + 17, rect_pos[1] + 17),
                                       color=self.instances_to_color[i],
                                       fill=self.instances_to_color[i])  # 绘制带颜色的矩形
        dpg.pop_container_stack()

    def update(self, projection, view):

        model = dpg.create_translation_matrix(self.pos) \
                * dpg.create_rotation_matrix(self.rot[0], [1, 0, 0]) \
                * dpg.create_rotation_matrix(self.rot[1], [0, 1, 0]) \
                * dpg.create_rotation_matrix(self.rot[2], [0, 0, 1]) \
                * dpg.create_scale_matrix(self.scale)
        # print(self.scale)
        # print(dpg.create_scale_matrix(self.scale))
        dpg.apply_transform(self.node, projection * view * model)
