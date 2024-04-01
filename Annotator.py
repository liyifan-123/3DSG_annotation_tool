import dearpygui.dearpygui as dpg
import math
import numpy as np
import torch
import trimesh
import os
import json
from utils import *
from graphviz import Digraph


class Annotator:

    def __init__(self, instances, relation_names, object_names, instances_to_label, scene_id, current_folder):
        self.scene_id = scene_id
        self.instances = instances
        self.instances_names = np.unique(instances)
        self.instances_to_label = instances_to_label
        self.relation_names = relation_names
        self.object_names = object_names
        self._initial_windows()
        self.current_folder = current_folder

        self.point_cloud = None
        self.sub = None
        self.obj = None
        self.pre = None
        self.annotation_result = init_annotation_result(self.instances_to_label, scene_id,
                                                        os.path.join(self.current_folder, "triplet_annotation.json"),
                                                        self.object_names)
        self.selected_triplet = None  # 存放triplet展示列表中即将被删除的triplet
        self.annotation_list_box_id = None
        self.annotate_relation_list = None
        self.to_add_relation = None  # 存放即将add的relation
        self.to_del_relation = None
        self.relation_list_box_id = None
        self.subject_combo = None
        self.pretrain_model_result = torch.load(r"data/ScanNet_sets/scannet_result_new.pth")

    def initialize(self, instances, instances_to_label, file_folder):
        self.save_annotation_result()
        self.instances = instances
        self.instances_names = self.point_cloud.instances_names
        self.instances_to_label = instances_to_label
        self.scene_id = self.scene_id
        self.current_folder = file_folder
        self.configure_items()

    def _show_recommend_result(self):
        if self.sub is not None and self.obj is not None:
            re = self.pretrain_model_result[self.scene_id]
            sub_id = int(self.sub.split("-")[0])
            obj_id = int(self.obj.split("-")[0])
            sub_i = re["instance_list"][0].index(sub_id)
            obj_i = re["instance_list"][0].index(obj_id)
            edge_id = (re["edge_indices"] == torch.tensor([sub_i, obj_i])).all(dim=1)
            edge_sort_logit = re["relation_logit"][edge_id].squeeze(0)
            edge_sort_idx = re["relation_sort_idx"][edge_id].squeeze(0)

            if edge_sort_logit[0] < 50:
                change_list = ["none_50.00"]  # 当最推荐的关系可能性小于50时
            else:
                change_list = []

            change_list += ["%s_%.2f" % (self.relation_names[edge_sort_idx[i] + 1], edge_sort_logit[i]) for i in
                            range(len(edge_sort_logit))]

            dpg.configure_item("recommend_rel_list", items=change_list)

    def _show_recommend_object(self):
        if self.sub is not None:
            sub_id = int(self.sub.split("-")[0])
            change_list_ins = self.point_cloud.distance_dict[sub_id]
            change_list = [f"{ins}-{self.object_names[self.instances_to_label[ins] - 1]}" for ins in change_list_ins]
            dpg.configure_item("recommend_obj_list", items=change_list)

    def _high_light(self, app_data, s_or_o):
        # 高亮场景中的物体
        instance = int(app_data.split("-")[0])
        self.point_cloud.highlight(instance, s_or_o)

    def _set_sub_obj_pre(self, sender, app_data, user_data):
        if sender == "subject":
            self.sub = app_data
            self._show_recommend_object()
            self._show_recommend_result()
            self._high_light(app_data, "sub")
        elif sender == "object":
            self.obj = app_data
            self._show_recommend_result()
            self._high_light(app_data, "obj")
        elif sender == "predicate":
            self.pre = app_data

    def _set_object(self, sender, app_data):
        # 在候选object列表中选择object
        dpg.configure_item("object", default_value=app_data)
        self.obj = app_data
        self._show_recommend_result()
        self._high_light(app_data, "obj")

    def _set_predicate(self, sender, app_data):
        # 在候选object列表中选择object
        print(app_data)
        dpg.configure_item("predicate", default_value=app_data.split("_")[0])
        self.pre = app_data.split("_")[0]

    def _set_selected_triplet(self, s, v, a):
        print(s, v, a)
        self.selected_triplet = int(v.split(":")[0])

    def _set_annotation_list(self):
        if len(self.annotation_result) != 0:
            dpg.configure_item(self.annotation_list_box_id,
                               items=[self._list_to_triplet(idx, i) for idx, i in
                                      enumerate(self.annotation_result["relationships"])])
        else:
            dpg.configure_item(self.annotation_list_box_id, item=None)

    def _delete_selected_triplet(self, s, v, a):
        print(s, v, a)
        del self.annotation_result["relationships"][self.selected_triplet]
        self._set_annotation_list()

    def _list_to_triplet(self, idx, l):
        sub = f"{l[0]}-{self.object_names[self.instances_to_label[l[0]] - 1]}"
        obj = f"{l[1]}-{self.object_names[self.instances_to_label[l[1]] - 1]}"
        return f"{idx}:{sub} {l[-1]} {obj}"

    def _get_add_rel(self, s, v):
        self.to_add_relation = v

    def _set_relation_list(self):
        if len(self.relation_names) != 0:
            dpg.configure_item(self.relation_list_box_id,
                               items=self.relation_names)
            dpg.configure_item(self.annotate_relation_list,
                               items=self.relation_names)
        else:
            dpg.configure_item(self.relation_list_box_id, item=None)

    def _add_relation(self, s, v):
        print(s, v)
        if self.to_add_relation not in self.relation_names:
            self.relation_names.append(self.to_add_relation)
        self._set_relation_list()

    def _set_rel_to_del(self, s, v, a):
        print(s, v, a)
        self.to_del_relation = v

    def _del_relation(self, s, v):
        print(s, v)
        del self.relation_names[self.relation_names.index(self.to_del_relation)]
        self._set_relation_list()

    def add_annotation_result(self):
        print(self.sub, self.pre, self.obj)
        sub_id = int(self.sub.split("-")[0])
        obj_id = int(self.obj.split("-")[0])
        t = [sub_id, obj_id, self.relation_names.index(self.pre), self.pre]
        if t not in self.annotation_result["relationships"]:
            self.annotation_result["relationships"].append(t)
            self._set_annotation_list()

    def save_annotation_result(self):
        filename = os.path.join(self.current_folder, "triplet_annotation.json")
        with open(filename, "w") as f:
            json.dump(self.annotation_result, f, indent=4)

        rel_file = "data/ScanNet_sets/relation_label.txt"
        with open(rel_file, "w") as f:
            for i in self.relation_names:
                f.write(i + "\n")

    def configure_items(self):
        dpg.delete_item("window_anno_0", children_only=True)
        dpg.delete_item("window_anno_1", children_only=True)
        dpg.delete_item("window_anno_2", children_only=True)
        dpg.delete_item("window_anno_3", children_only=True)
        dpg.delete_item("window_anno_4", children_only=True)
        self.annotation_result = init_annotation_result(self.instances_to_label, self.scene_id,
                                                        f"{self.scene_id}_annotation.json", self.object_names)
        self.show_controls()

    def _initial_windows(self):
        self.window_anno_0 = dpg.add_window(label="Annotation", width=500, height=500, tag="window_anno_0")
        self.window_anno_1 = dpg.add_window(label="Annotation result", width=500, height=1000, tag="window_anno_1")
        self.window_anno_2 = dpg.add_window(label="Relation Backup", width=1500, height=800, tag="window_anno_2")
        self.window_anno_3 = dpg.add_window(label="recommend_relation", width=1500, height=800, tag="window_anno_3")
        self.window_anno_4 = dpg.add_window(label="recommend_object", width=1500, height=800, tag="window_anno_4")

    def show_scene_graph(self):
        scene_graph = Digraph(comment='Scene Graph', format='png')
        node_list = []
        triplet_dict = {}
        for idx, tri in enumerate(self.annotation_result["relationships"]):
            if tri[-1] == "none":
                continue
            sub = f"{tri[0]}-{self.object_names[self.instances_to_label[tri[0]] - 1]}"
            obj = f"{tri[1]}-{self.object_names[self.instances_to_label[tri[1]] - 1]}"
            if (sub, obj) not in triplet_dict.keys():
                triplet_dict[(sub, obj)] = [tri[-1]]
            else:
                triplet_dict[(sub, obj)].append(tri[-1])

        for key, value in triplet_dict.items():
            if key[0] not in node_list:
                scene_graph.node(key[0], key[0], color="#000000", fillcolor="#f5d0e1",
                                 fontcolor="#000000", shape="box", style="filled")
                node_list.append(key[0])
            if key[1] not in node_list:
                scene_graph.node(key[1], key[1], color="#000000", fillcolor="#f5d0e1",
                                 fontcolor="#000000", shape="box", style="filled")
                node_list.append(key[1])
            scene_graph.edge(key[0], key[1], color="#000000", label=",".join(value))

        scene_graph.render("temp.gv", view=True)

    def show_controls(self):
        dpg.push_container_stack(self.window_anno_0)
        dpg.add_text(tag="scene_text", default_value=self.scene_id)
        dpg.add_combo([f"{i}-{self.object_names[self.instances_to_label[i] - 1]}" for i in self.instances_names],
                      tag="subject", label="subject", default_value=" ", callback=self._set_sub_obj_pre)
        self.annotate_relation_list = dpg.add_combo(self.relation_names, tag="predicate",
                                                    label="predicate", default_value=" ",
                                                    callback=self._set_sub_obj_pre)
        dpg.add_combo([f"{i}-{self.object_names[self.instances_to_label[i] - 1]}" for i in self.instances_names],
                      tag="object",
                      label="object", default_value=" ", callback=self._set_sub_obj_pre)
        dpg.add_button(label="Add", callback=self.add_annotation_result)
        dpg.pop_container_stack()

        dpg.push_container_stack(self.window_anno_1)
        self.annotation_list_box_id = \
            dpg.add_listbox(
                [self._list_to_triplet(idx, i) for idx, i in enumerate(self.annotation_result["relationships"])]
                , callback=self._set_selected_triplet, num_items=10, width=300)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Delete", callback=self._delete_selected_triplet)
            dpg.add_button(label="Show Scene Graph", callback=self.show_scene_graph)
        dpg.pop_container_stack()

        dpg.push_container_stack(self.window_anno_2)
        dpg.add_input_text(label="input text", default_value="New_relation", callback=self._get_add_rel)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Add", callback=self._add_relation)
            dpg.add_button(label="Delete", callback=self._del_relation)

        self.relation_list_box_id = \
            dpg.add_listbox(
                self.relation_names,
                label="listbox", callback=self._set_rel_to_del, num_items=10)
        dpg.pop_container_stack()

        dpg.push_container_stack(self.window_anno_3)
        dpg.add_text("recommend_relations")
        dpg.add_listbox([" " for i in range(10)], num_items=10, tag="recommend_rel_list", callback=self._set_predicate)
        dpg.pop_container_stack()

        dpg.push_container_stack(self.window_anno_4)
        dpg.add_text("recommend_objects")
        dpg.add_listbox([" " for i in range(10)], num_items=10, tag="recommend_obj_list", callback=self._set_object)
        dpg.pop_container_stack()
