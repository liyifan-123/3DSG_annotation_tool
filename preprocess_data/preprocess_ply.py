import json
import os
from tqdm import tqdm
import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
import pandas as pd


def read_object_label_list():
    file = "scannetv2-labels.combined.tsv"
    df = pd.read_csv(file, delimiter='\t')
    label_raw = df["raw_category"]
    label_fixed = np.array(df["category"])
    label_dict = {i: label_fixed[idx] for idx, i in enumerate(label_raw)}
    return np.unique(label_fixed), label_dict


def read_labels(plydata):
    data = plydata.metadata['_ply_raw']['vertex']['data']
    try:
        labels = data['objectId']
    except:
        labels = data['label']
    return labels


def load_mesh(label_file="scene0000_01_vh_clean_2.labels.ply"):
    plydata = trimesh.load(label_file, process=False)
    points = np.array(plydata.vertices)
    instances = read_labels(plydata).flatten()

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

    radis = np.max(np.linalg.norm(points - center, axis=-1))

    return points, rgbs, instances, radis


if __name__ == "__main__":
    root_dir = "D:\Annotation System\data"
    folder_list = os.listdir(root_dir)
    if "ScanNet_sets" in folder_list:
        folder_list.remove("ScanNet_sets")
    label_fixed, label_dict = read_object_label_list()

    for scene_id in tqdm(folder_list):
        label_file = os.path.join(root_dir, scene_id, f"{scene_id}_vh_clean_2.labels.ply")
        agg_file = os.path.join(root_dir, scene_id, f"{scene_id}_vh_clean_2.0.010000.segs.json")
        seg_file = os.path.join(root_dir, scene_id, f"{scene_id}_vh_clean.aggregation.json")

        with open(label_file, 'rb') as f:
            plydata = PlyData.read(f)

        points = plydata["vertex"]["x"]

        with open(agg_file, "r") as f:
            agg_dict = json.load(f)
            agg_list = np.array(agg_dict["segIndices"])

        with open(seg_file, "r") as f:
            seg_dict = json.load(f)

        instances = np.zeros(points.shape[0])
        fixed_labels = np.zeros(points.shape[0])  # 存储fixed label的index

        for object_dict in seg_dict["segGroups"]:
            id = object_dict["objectId"]
            label_t = object_dict["label"]
            segments = object_dict["segments"]
            for seg in segments:
                t = np.where(agg_list == seg)
                instances[t] = id
                fixed_labels[t] = np.where(label_fixed == label_dict[label_t])[0].item()

        vertex_data = plydata['vertex'].data
        new_vertex_data = []
        for i, v in enumerate(vertex_data):
            new_v = list(v) + [instances[i], fixed_labels[i]]
            new_vertex_data.append(tuple(new_v))

        # 定义新的顶点元素，包括新的属性
        new_vertex_element = PlyElement.describe(
            np.array(new_vertex_data, dtype=[*vertex_data.dtype.descr, ('objectId', 'i4'), ("label_raw", "i4")]),
            'vertex')

        # 创建新的PlyData实例
        new_plydata = PlyData([new_vertex_element, plydata.elements[1]], text=plydata.text)

        with open(os.path.join(root_dir, scene_id, f'{scene_id}_vh_clean_2.labels.instances.ply'), 'wb') as f:
            new_plydata.write(f)
