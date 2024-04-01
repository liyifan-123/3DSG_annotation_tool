import numpy as np
import torch


def read_text_class(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
        data = [i.strip() for i in data]
    return data


if __name__ == "__main__":
    file_path = r"D:\Annotation System\data\ScanNet_sets\scannet_result.pth"
    result = torch.load(file_path)
    object_names = read_text_class(r"D:\Annotation System\data\ScanNet_sets\object_fixed_label.txt")
    relation_names = read_text_class(r"D:\Annotation System\data\ScanNet_sets\relation_label.txt")
    new_result = {}
    for key, v in result.items():
        relation_logit, sort_idx = torch.sort(v["relation_logit"], dim=1, descending=True)
        v["relation_logit"] = relation_logit * 100
        v["relation_sort_idx"] = sort_idx
        new_result[key] = v
    torch.save(new_result, f"D:\Annotation System\data\ScanNet_sets\scannet_result_new.pth")
