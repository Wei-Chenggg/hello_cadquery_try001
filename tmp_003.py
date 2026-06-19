from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from toolbox.common_functions import *
STP_FOLDER = Path("./stp_files_basic_hole")



def split_params(data):
    ret, buf = [], []
    layer, in_text, i = 0, False, 0
    while i < len(data):
        ch = data[i]
        if ch == "'":
            buf.append(ch)
            if i + 1 < len(data) and data[i + 1] == "'":
                buf.append(data[i + 1]); i += 2; continue
            in_text = not in_text
        elif not in_text and ch in "()":
            layer += 1 if ch == "(" else -1
            buf.append(ch)
        elif not in_text and ch == "," and layer == 0:
            ret.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
        i += 1
    if buf: ret.append("".join(buf).strip())
    return ret
def parse_stp_entities(stp_path):
    txt_data = Path(stp_path).read_text(encoding="utf-8", errors="ignore")
    entities = {}
    for m in re.finditer(r"#(\d+)\s*=\s*(.*?);", txt_data, re.S):
        id_name = "#" + m.group(1)
        text = " ".join(m.group(2).split())
        f = re.match(r"([A-Z0-9_]+)\s*\((.*)\)$", text)
        func, items = (f.group(1), split_params(f.group(2))) if f else ("", [])
        entities[id_name] = {"id": id_name, "idx": int(m.group(1)), "func": func, "items": items, "refs": re.findall(r"#\d+", text), "text": text}
    return entities
def get_unique_stp_func(entities):
    funcs = [entities[k]["func"] for k in entities]
    unique_funcs = np.unique(funcs)
    print(unique_funcs)
    return unique_funcs
def get_tree_relation(entities, id_name, max_level=20, _used=None):
    _used = set() if _used is None else set(_used)
    if id_name not in entities or id_name in _used or max_level <= 0:
        return {id_name: None}
    _used.add(id_name)
    refs = entities[id_name].get("refs", [])
    if not refs:
        return {id_name: None}
    data = {}
    for ref in refs:
        item = get_tree_relation(entities, ref, max_level - 1, _used)
        data[ref] = item.get(ref)
    return {id_name: data}
def gather_circles(entities):
    circles = {}
    for this_id in sorted(entities, key=lambda k: entities[k]["idx"]):
        item = entities[this_id]
        if item["func"] != "CIRCLE":   continue
        refs = item.get("refs", [])
        if not refs:  continue
        axis_id = refs[0]
        axis_item = entities.get(axis_id, {})
        if axis_item.get("func") != "AXIS2_PLACEMENT_3D": continue
        axis_refs = axis_item.get("refs", [])
        org = _point_or_dir(axis_refs[0]) if len(axis_refs) > 0 else None
        dir_n = _point_or_dir(axis_refs[1]) if len(axis_refs) > 1 else None
        dir_x = _point_or_dir(axis_refs[2]) if len(axis_refs) > 2 else None
        radius = _stp_words_to_nums(item["items"][-1] if item.get("items") else item.get("text", ""))
        circles[this_id] = {"org": org, "dir_n": dir_n, "dir_x": dir_x, "r": radius[0] if radius else None}
    return circles
def _stp_words_to_nums(data):
    return [float(x) for x in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", data or "")]
def _point_or_dir(id_name, entity_data=None):
    entity_data = globals().get("entities", {}) if entity_data is None else entity_data
    item = entity_data.get(id_name, {})
    if item.get("func") not in ("CARTESIAN_POINT", "DIRECTION"):
        return None
    nums = _stp_words_to_nums(item["items"][-1] if item.get("items") else item.get("text", ""))
    return nums if nums else None

def gather_surface_edges(entities):
    surface_edges = {}
    for face_id in sorted(entities, key=lambda k: entities[k]["idx"]):
        face_item = entities[face_id]
        if face_item["func"] != "ADVANCED_FACE": continue
        if len(face_item.get("items", [])) < 3: continue
        bound_ids = re.findall(r"#\d+", face_item["items"][1])
        surface_ids = re.findall(r"#\d+", face_item["items"][2])
        if not surface_ids: continue
        surface_id = surface_ids[0]
        for bound_id in bound_ids:
            bound_item = entities.get(bound_id, {})
            loop_ids = bound_item.get("refs", [])
            if not loop_ids: continue
            loop_item = entities.get(loop_ids[0], {})
            oriented_ids = re.findall(r"#\d+", loop_item["items"][-1]) if loop_item.get("items") else []
            for oriented_id in oriented_ids:
                oriented_item = entities.get(oriented_id, {})
                edge_ids = [ref for ref in oriented_item.get("refs", []) if entities.get(ref, {}).get("func") == "EDGE_CURVE"]
                for edge_id in edge_ids:
                    edge_item = entities.get(edge_id, {})
                    refs = edge_item.get("refs", [])
                    if len(refs) < 3: continue
                    curve_id = refs[2]
                    curve_item = entities.get(curve_id, {})
                    if curve_item.get("func") in ("SURFACE_CURVE", "SEAM_CURVE"):
                        curve_refs = curve_item.get("refs", [])
                        if curve_refs: curve_id = curve_refs[0]
                        curve_item = entities.get(curve_id, {})
                    if curve_item.get("func") not in ("CIRCLE", "LINE"): continue
                    surface_edges.setdefault(surface_id, [])
                    if curve_id not in surface_edges[surface_id]:
                        surface_edges[surface_id].append(curve_id)
    return surface_edges
def gather_plane(entities):
    surface_edges = gather_surface_edges(entities)
    plane = {}
    for this_id in sorted(entities, key=lambda k: entities[k]["idx"]):
        item = entities[this_id]
        if item["func"] != "PLANE": continue
        refs = item.get("refs", [])
        axis_item = entities.get(refs[0], {}) if refs else {}
        axis_refs = axis_item.get("refs", [])
        org = _point_or_dir(axis_refs[0], entities) if len(axis_refs) > 0 else None
        dir_n = _point_or_dir(axis_refs[1], entities) if len(axis_refs) > 1 else None
        dir_x = _point_or_dir(axis_refs[2], entities) if len(axis_refs) > 2 else None
        if org is not None and dir_n is not None:
            org_arr = np.array(org, dtype=float)
            dir_n_arr = np.array(dir_n, dtype=float)
            dir_n_len = np.linalg.norm(dir_n_arr)
            height = float(np.dot(org_arr, dir_n_arr / dir_n_len)) if dir_n_len > 0 else None
        else:
            height = None
        plane[this_id] = {"org": org, "dir_n": dir_n, "dir_x": dir_x, "height": height, "dist": abs(height) if height is not None else None, "edges": surface_edges.get(this_id, [])}
        
        print(plane[this_id])
    return plane
def gather_cylinder(entities):
    surface_edges = gather_surface_edges(entities)
    cylinder = {}
    for face_id in sorted(entities, key=lambda k: entities[k]["idx"]):
        face_item = entities[face_id]
        if face_item["func"] != "ADVANCED_FACE": continue
        if len(face_item.get("items", [])) < 3: continue
        bound_ids = re.findall(r"#\d+", face_item["items"][1])
        surface_ids = re.findall(r"#\d+", face_item["items"][2])
        if not surface_ids: continue
        surface_id = surface_ids[0]
        surface_item = entities.get(surface_id, {})
        if surface_item.get("func") != "CYLINDRICAL_SURFACE": continue
        refs = surface_item.get("refs", [])
        axis_item = entities.get(refs[0], {}) if refs else {}
        axis_refs = axis_item.get("refs", [])
        org = _point_or_dir(axis_refs[0], entities) if len(axis_refs) > 0 else None
        dir_n = _point_or_dir(axis_refs[1], entities) if len(axis_refs) > 1 else None
        dir_x = _point_or_dir(axis_refs[2], entities) if len(axis_refs) > 2 else None
        radius = _stp_words_to_nums(surface_item["items"][-1] if surface_item.get("items") else surface_item.get("text", ""))
        cylinder[face_id] = {"face_id": face_id, "surface_id": surface_id, "type": "CYLINDER", "org": org, "dir_n": dir_n, "dir_x": dir_x, "r": radius[0] if radius else None, "face_same_sense": face_item["items"][3] if len(face_item.get("items", [])) > 3 else None, "edges": surface_edges.get(surface_id, [])}
        print(cylinder[face_id])
    return cylinder


fnames, fpaths = get_files("./stp_files_basic_hole", ["stp"])
for i, fpath in enumerate(fpaths):
    fname = fnames[i]
    entities = parse_stp_entities(fpath)
    circles = gather_circles(entities)
    get_unique_stp_func(entities)
    gather_cylinder(entities)
    print("---------")
    gather_plane(entities)
    
    # print()
    # print(circles)
    # data = flatten_faces(entities)
    # print(data)
    # ret = get_unique_stp_func(entities)
    # ret = get_tree_relation(entities)
    # a=bb
