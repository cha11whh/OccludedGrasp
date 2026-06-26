import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]


def _resolve_unobench_syn_root():
    candidates = [
        REPO_ROOT.parent / "Unobench" / "UnoBenchSyn",
        REPO_ROOT.parent / "UnoBench" / "UnoBenchSyn",
        REPO_ROOT / "UnoBenchSyn",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


UNOBENCH_SYN_ROOT = _resolve_unobench_syn_root()

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

######################################################
# Train som
# UNOGRASP_SOM_BASE = {
#     "annotation_path": str(UNOBENCH_SYN_ROOT / "training_sft_set" / "som" / "split_scene_base" / "train.json"),
#     "data_path": str(UNOBENCH_SYN_ROOT),
# }

# UNOGRASP_SOM_VAL_BASE = {
#     "annotation_path": str(UNOBENCH_SYN_ROOT / "training_sft_set" / "som" / "split_scene_base" / "val.json"),
#     "data_path": str(UNOBENCH_SYN_ROOT),
# }

# ratio
UNOGRASP_SOM_RATIO = {
    "annotation_path": str(UNOBENCH_SYN_ROOT / "training_sft_set" / "som" / "split_scene_ratio_only" / "train.json"),
    "data_path": str(UNOBENCH_SYN_ROOT),
}
UNOGRASP_SOM_VAL_RATIO = {
    "annotation_path": str(UNOBENCH_SYN_ROOT / "training_sft_set" / "som" / "split_scene_ratio_only" / "val.json"),
    "data_path": str(UNOBENCH_SYN_ROOT),
}

UNOGRASP_SOM_RATIO_SMALL = {
    "annotation_path": str(UNOBENCH_SYN_ROOT / "training_sft_set_small" / "som" / "split_scene_ratio_only" / "train_small.json"),
    "data_path": str(UNOBENCH_SYN_ROOT),
}
##########################################################################################################################################
# Train nlp
# UNOGRASP_NLP_BASE = {
#     "annotation_path": str(UNOBENCH_SYN_ROOT / "training_sft_set" / "nlp" / "split_scene_base" / "train.json"),
#     "data_path": str(UNOBENCH_SYN_ROOT),
# }

UNOGRASP_NLP_RATIO = {
    "annotation_path": str(UNOBENCH_SYN_ROOT / "training_sft_set" / "nlp" / "split_scene_ratio_only" / "train.json"),
    "data_path": str(UNOBENCH_SYN_ROOT),
}

UNOGRASP_NLP_RATIO_SMALL = {
    "annotation_path": str(UNOBENCH_SYN_ROOT / "training_sft_set_small" / "nlp" / "split_scene_ratio_only" / "train_small.json"),
    "data_path": str(UNOBENCH_SYN_ROOT),
}

data_dict = {
    # "unograsp_som_base": UNOGRASP_SOM_BASE,
    # "unograsp_som_val_base": UNOGRASP_SOM_VAL_BASE,
    "unograsp_som_ratio": UNOGRASP_SOM_RATIO,
    "unograsp_som_ratio_small": UNOGRASP_SOM_RATIO_SMALL,
    # "unograsp_som_val_ratio": UNOGRASP_SOM_VAL_RATIO,
    # "unograsp_nlp_base": UNOGRASP_NLP_BASE,
    "unograsp_nlp_ratio": UNOGRASP_NLP_RATIO,
    "unograsp_nlp_ratio_small": UNOGRASP_NLP_RATIO_SMALL,
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
