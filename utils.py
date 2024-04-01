import os.path

import trimesh
import numpy as np
import json
from plyfile import PlyData


def read_labels(plydata, label_type):
    data = plydata.metadata['_ply_raw']['vertex']['data']
    instances = data['objectId']
    if label_type == "raw":
        labels = data["label_raw"] + 1
    else:
        labels = data["label"]
    return instances.flatten(), labels.flatten()


def load_mesh(label_file="scene0000_01_vh_clean_2.labels.instances.ply", label_type="raw"):
    plydata = trimesh.load(label_file, process=False)
    points = np.array(plydata.vertices)
    instances, labels = read_labels(plydata, label_type)

    # min_point = np.min(points)
    #
    # points += -min_point

    rgbs = np.array(plydata.visual.vertex_colors.tolist())[:, :3]
    rgbs = np.hstack([rgbs, np.ones([rgbs.shape[0], 1]) * 255])
    rgbs = rgbs.astype("int").tolist()

    center = (np.amax(points, axis=0) + np.amin(points, axis=0)) / 2

    points[:, 0] = points[:, 0] - center[0]
    points[:, 1] = points[:, 1] - center[1]
    points[:, 2] = points[:, 2] - center[2]

    with open(label_file, 'rb') as f:
        plydata = PlyData.read(label_file)

    face = plydata["face"]["vertex_indices"]

    return points, rgbs, instances, labels, face


def read_text_class(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
        data = [i.strip() for i in data]
    return data


def get_instance_color_label_dict(instances, colors, labels):
    pairs = zip(instances, colors)
    D0 = {key: value for key, value in pairs}
    pairs = zip(instances, labels)
    D1 = {key: value for key, value in pairs}
    return D0, D1


def init_annotation_result(instances_to_label, scene_id, file, object_names):
    if os.path.exists(file):
        with open(file, "r") as f:
            r = json.load(f)
    else:
        r = {}
    if "relationships" not in r.keys():
        r["relationships"] = []
    if "scene_id" not in r.keys():
        r["scene_id"] = scene_id
    if "objects" not in r.keys():
        r["objects"] = {str(key): object_names[value - 1] for key, value in instances_to_label.items()}

    return r


def get_bbox(t_p):
    l_d = np.min(t_p, axis=0)  # 左下角
    r_u = np.max(t_p, axis=0)  # 右上角
    a_1 = np.array([l_d[0], l_d[1], r_u[2]])
    a_2 = np.array([r_u[0], l_d[1], r_u[2]])
    a_3 = np.array([l_d[0], r_u[1], r_u[2]])
    a_6 = np.array([r_u[0], l_d[1], l_d[2]])
    a_7 = np.array([l_d[0], r_u[1], l_d[2]])
    a_8 = np.array([r_u[0], r_u[1], l_d[2]])

    c_0 = np.array([(r_u[0] + l_d[0]) / 2, (r_u[1] + l_d[1]) / 2, l_d[2]])
    c_1 = np.array([(r_u[0] + l_d[0]) / 2, (r_u[1] + l_d[1]) / 2, r_u[2]])
    c_2 = np.array([(r_u[0] + l_d[0]) / 2, l_d[1], (r_u[2] + l_d[2]) / 2])
    c_3 = np.array([(r_u[0] + l_d[0]) / 2, r_u[1], (r_u[2] + l_d[2]) / 2])
    c_4 = np.array([l_d[0], (r_u[1] + l_d[1]) / 2, (r_u[2] + l_d[2]) / 2])
    c_5 = np.array([r_u[0], (r_u[1] + l_d[1]) / 2, (r_u[2] + l_d[2]) / 2])
    c = np.array([(r_u[0] + l_d[0]) / 2, (r_u[1] + l_d[1]) / 2, (r_u[2] + l_d[2]) / 2])
    return [c, a_1, a_2, a_3, r_u, l_d, a_6, a_7, a_8, c_0, c_1, c_2, c_3, c_4, c_5]


def get_distance_dict(points, instances, instance_list, instance_to_labels, object_names, radis):
    point_dict = {}
    for ins in instance_list:
        index = np.where(instances == ins)
        t_p = points[index]

        # point_dict[ins] = [c, l_d, r_u, a_1, a_2, a_3, a_6, a_7, a_8, c_0, c_1, c_2, c_3, c_4, c_5]
        point_dict[ins] = get_bbox(t_p)

    closet = {}
    for sub in instance_list:
        dis_list = []
        for obj in instance_list:
            min_dis = float('inf')
            if sub == obj:
                dis_list.append(min_dis)
                continue
            for point1 in point_dict[sub]:
                for point2 in point_dict[obj]:
                    dis = np.linalg.norm(point1 - point2)
                    if dis < min_dis:
                        min_dis = dis
            dis_list.append(min_dis)
        dis_list = np.array(dis_list)
        index = np.where(dis_list < radis)[0]
        closet[sub] = instance_list[index]

    return closet
