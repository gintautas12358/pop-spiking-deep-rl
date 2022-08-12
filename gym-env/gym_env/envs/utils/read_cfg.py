import yaml
from yaml.loader import SafeLoader
import numpy as np

yaml_path = "./kuka/cfg/cfg.yaml"

def read_cfg():
    cfg = None
    with open(yaml_path) as f:
        cfg = yaml.load(f, Loader=SafeLoader)
    
    return cfg

def get_jposes():
    cfg = read_cfg()
    jposes_raw = cfg["poses"]["joint"]
    jposes = {}
    for pose_k, pose_val in jposes_raw.items():
        jposes[pose_k] = np.array([np.deg2rad(x) for x in pose_val])

    return jposes

def get_cposes():
    cfg = read_cfg()
    cposes_raw = cfg["poses"]["cartesian"]
    cposes = {}
    for pose_k, pose_val in cposes_raw.items():
        cposes[pose_k] = np.array(pose_val[:3] + [np.deg2rad(x) for x in pose_val[3:]])

    return cposes

def get_jerr_lim():
    cfg = read_cfg()
    return cfg["jerr_limit"]

def get_cerr_lim():
    cfg = read_cfg()
    return cfg["cerr_limit"]

def get_mjc_xml():
    cfg = read_cfg()
    return cfg["mujoco_model_xml"]