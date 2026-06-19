from pathlib import Path
import re
import numpy as np
import cadquery as cq
from cadquery import importers, Compound
from cadquery.occ_impl.exporters.dxf import DxfDocument
from OCP.BRepLib import BRepLib
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from toolbox.common_functions import get_files

STP_FOLDER = Path("./stp_files_basic_hole")
SAVE_PATH = STP_FOLDER / "quick_view_004.dxf"
dxf_color_map = {"白": 7, "綠": 3, "青": 4}
dxf_layer_items = {
    "0": {"color": dxf_color_map["白"], "linetype": "CONTINUOUS", "lineweight": 18},
    "HID": {"color": dxf_color_map["綠"], "linetype": "HIDDEN", "lineweight": 18},
    "TAN": {"color": dxf_color_map["青"], "linetype": "CONTINUOUS", "lineweight": 18},
}
VIEW_CFGS = {
    "top": {"N": (0, 1, 0), "U": (0, 0, -1), "grid": (1, 2)},
    "left": {"N": (-1, 0, 0), "U": (0, 1, 0), "grid": (0, 1)},
    "front": {"N": (0, 0, 1), "U": (0, 1, 0), "grid": (1, 1)},
    "right": {"N": (1, 0, 0), "U": (0, 1, 0), "grid": (2, 1)},
    "bottom": {"N": (0, -1, 0), "U": (0, 0, 1), "grid": (1, 0)},
}
def sort_key(path):
    path = Path(path)
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else path.stem

class STP_Tmp004:
    def __init__(self):
        self.dxf_doc = None
        self.doc = None
        self.stp_path = None
        self.model = None
        self.shape = None
        self.entities = {}
        self.gather_data = {}
    def init_dxf_document(self):
        dxf_doc = DxfDocument(setup=["linetypes"])
        if "HIDDEN" not in dxf_doc.document.linetypes:
            dxf_doc.document.linetypes.add("HIDDEN", pattern=[4, 3, -1], description="Hidden __ __ __ __ __ __ __")
        for layer_name, layer_item in dxf_layer_items.items():
            if layer_name not in dxf_doc.document.layers:
                dxf_doc.add_layer(layer_name, color=layer_item["color"], linetype=layer_item["linetype"])
            layer = dxf_doc.document.layers.get(layer_name)
            layer.dxf.color = layer_item["color"]
            layer.dxf.linetype = layer_item["linetype"]
            layer.dxf.lineweight = layer_item["lineweight"]
        self.dxf_doc = dxf_doc
        self.doc = dxf_doc.document
        return dxf_doc
    def split_params(self, data):
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
    def load_stp(self, stp_path):
        self.stp_path = Path(stp_path)
        self.model = importers.importStep(str(self.stp_path))
        self.shape = self.model.val()
        txt_data = self.stp_path.read_text(encoding="utf-8", errors="ignore")
        entities = {}
        for m in re.finditer(r"#(\d+)\s*=\s*(.*?);", txt_data, re.S):
            id_name = "#" + m.group(1)
            text = " ".join(m.group(2).split())
            f = re.match(r"([A-Z0-9_]+)\s*\((.*)\)$", text)
            func, items = (f.group(1), self.split_params(f.group(2))) if f else ("", [])
            entities[id_name] = {"id": id_name, "idx": int(m.group(1)), "func": func, "items": items, "refs": re.findall(r"#\d+", text), "text": text}
        self.entities = entities
        return entities
    def _stp_words_to_nums(self, data):
        return [float(x) for x in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", data or "")]
    def _point_or_dir(self, id_name):
        item = self.entities.get(id_name, {})
        if item.get("func") not in ("CARTESIAN_POINT", "DIRECTION"):
            return None
        nums = self._stp_words_to_nums(item["items"][-1] if item.get("items") else item.get("text", ""))
        return nums if nums else None
    def gather_circles(self):
        circles = {}
        for this_id in sorted(self.entities, key=lambda k: self.entities[k]["idx"]):
            item = self.entities[this_id]
            if item["func"] != "CIRCLE": continue
            refs = item.get("refs", [])
            if not refs: continue
            axis_item = self.entities.get(refs[0], {})
            if axis_item.get("func") != "AXIS2_PLACEMENT_3D": continue
            axis_refs = axis_item.get("refs", [])
            org = self._point_or_dir(axis_refs[0]) if len(axis_refs) > 0 else None
            dir_n = self._point_or_dir(axis_refs[1]) if len(axis_refs) > 1 else None
            dir_x = self._point_or_dir(axis_refs[2]) if len(axis_refs) > 2 else None
            radius = self._stp_words_to_nums(item["items"][-1] if item.get("items") else item.get("text", ""))
            circles[this_id] = {"org": org, "dir_n": dir_n, "dir_x": dir_x, "r": radius[0] if radius else None}
        return circles
    def gather_surface_edges(self):
        surface_edges = {}
        for face_id in sorted(self.entities, key=lambda k: self.entities[k]["idx"]):
            face_item = self.entities[face_id]
            if face_item["func"] != "ADVANCED_FACE": continue
            if len(face_item.get("items", [])) < 3: continue
            bound_ids = re.findall(r"#\d+", face_item["items"][1])
            surface_ids = re.findall(r"#\d+", face_item["items"][2])
            if not surface_ids: continue
            surface_id = surface_ids[0]
            for bound_id in bound_ids:
                bound_item = self.entities.get(bound_id, {})
                loop_ids = bound_item.get("refs", [])
                if not loop_ids: continue
                loop_item = self.entities.get(loop_ids[0], {})
                oriented_ids = re.findall(r"#\d+", loop_item["items"][-1]) if loop_item.get("items") else []
                for oriented_id in oriented_ids:
                    oriented_item = self.entities.get(oriented_id, {})
                    edge_ids = [ref for ref in oriented_item.get("refs", []) if self.entities.get(ref, {}).get("func") == "EDGE_CURVE"]
                    for edge_id in edge_ids:
                        edge_item = self.entities.get(edge_id, {})
                        refs = edge_item.get("refs", [])
                        if len(refs) < 3: continue
                        curve_id = refs[2]
                        curve_item = self.entities.get(curve_id, {})
                        if curve_item.get("func") in ("SURFACE_CURVE", "SEAM_CURVE"):
                            curve_refs = curve_item.get("refs", [])
                            if curve_refs: curve_id = curve_refs[0]
                            curve_item = self.entities.get(curve_id, {})
                        if curve_item.get("func") not in ("CIRCLE", "LINE"): continue
                        surface_edges.setdefault(surface_id, [])
                        if curve_id not in surface_edges[surface_id]:
                            surface_edges[surface_id].append(curve_id)
        return surface_edges
    def gather_plane(self):
        surface_edges = self.gather_surface_edges()
        plane = {}
        for this_id in sorted(self.entities, key=lambda k: self.entities[k]["idx"]):
            item = self.entities[this_id]
            if item["func"] != "PLANE": continue
            refs = item.get("refs", [])
            axis_item = self.entities.get(refs[0], {}) if refs else {}
            axis_refs = axis_item.get("refs", [])
            org = self._point_or_dir(axis_refs[0]) if len(axis_refs) > 0 else None
            dir_n = self._point_or_dir(axis_refs[1]) if len(axis_refs) > 1 else None
            dir_x = self._point_or_dir(axis_refs[2]) if len(axis_refs) > 2 else None
            if org is not None and dir_n is not None:
                org_arr = np.array(org, dtype=float)
                dir_n_arr = np.array(dir_n, dtype=float)
                dir_n_len = np.linalg.norm(dir_n_arr)
                height = float(np.dot(org_arr, dir_n_arr / dir_n_len)) if dir_n_len > 0 else None
            else:
                height = None
            plane[this_id] = {"org": org, "dir_n": dir_n, "dir_x": dir_x, "height": height, "dist": abs(height) if height is not None else None, "edges": surface_edges.get(this_id, [])}
        return plane
    def gather_cylinder(self):
        return self._gather_axis_surface("CYLINDRICAL_SURFACE", "CYLINDER")
    def gather_cone(self):
        return self._gather_axis_surface("CONICAL_SURFACE", "CONE")
    def gather_torus(self):
        return self._gather_axis_surface("TOROIDAL_SURFACE", "TORUS")
    def _gather_axis_surface(self, func_name, type_name):
        surface_edges = self.gather_surface_edges()
        data = {}
        for face_id in sorted(self.entities, key=lambda k: self.entities[k]["idx"]):
            face_item = self.entities[face_id]
            if face_item["func"] != "ADVANCED_FACE": continue
            if len(face_item.get("items", [])) < 3: continue
            surface_ids = re.findall(r"#\d+", face_item["items"][2])
            if not surface_ids: continue
            surface_id = surface_ids[0]
            surface_item = self.entities.get(surface_id, {})
            if surface_item.get("func") != func_name: continue
            refs = surface_item.get("refs", [])
            axis_item = self.entities.get(refs[0], {}) if refs else {}
            axis_refs = axis_item.get("refs", [])
            org = self._point_or_dir(axis_refs[0]) if len(axis_refs) > 0 else None
            dir_n = self._point_or_dir(axis_refs[1]) if len(axis_refs) > 1 else None
            dir_x = self._point_or_dir(axis_refs[2]) if len(axis_refs) > 2 else None
            nums = self._stp_words_to_nums(",".join(surface_item.get("items", [])[-2:]) if len(surface_item.get("items", [])) >= 2 else surface_item.get("text", ""))
            item = {"face_id": face_id, "surface_id": surface_id, "type": type_name, "org": org, "dir_n": dir_n, "dir_x": dir_x, "face_same_sense": face_item["items"][3] if len(face_item.get("items", [])) > 3 else None, "edges": surface_edges.get(surface_id, [])}
            if func_name == "CYLINDRICAL_SURFACE":
                item["r"] = nums[0] if nums else None
            elif func_name == "CONICAL_SURFACE":
                item["r"] = nums[0] if nums else None
                item["semi_angle"] = nums[1] if len(nums) > 1 else None
            elif func_name == "TOROIDAL_SURFACE":
                item["major_r"] = nums[0] if nums else None
                item["minor_r"] = nums[1] if len(nums) > 1 else None
                item["r"] = item["minor_r"]
            data[face_id] = item
        return data
    def gather_all(self):
        self.gather_data = {
            "circles": self.gather_circles(),
            "surface_edges": self.gather_surface_edges(),
            "plane": self.gather_plane(),
            "cylinder": self.gather_cylinder(),
            "cone": self.gather_cone(),
            "torus": self.gather_torus(),
        }
        return self.gather_data
    def cal_proj_hlr(self, view_normal_dir, x_dir, eta=1e-3):
        hlr = HLRBRep_Algo()
        hlr.Add(self.shape.wrapped)
        hlr.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(), gp_Dir(*view_normal_dir), gp_Dir(*x_dir))))
        hlr.Update()
        hlr.Hide()
        hlr_shapes = HLRBRep_HLRToShape(hlr)
        visible = []
        hidden = []
        tan = []
        for target, items in (
            (visible, (hlr_shapes.VCompound(), hlr_shapes.OutLineVCompound())),
            (tan, (hlr_shapes.Rg1LineVCompound(),)),
            (hidden, (hlr_shapes.HCompound(), hlr_shapes.OutLineHCompound())),
        ):
            for item in items:
                if item.IsNull():
                    continue
                BRepLib.BuildCurves3d_s(item, eta)
                for edge in Compound(item).Edges():
                    target.append(edge)
        return visible, hidden, tan
    def get_x_dir(self, view):
        u = view["U"]
        n = view["N"]
        return (u[1] * n[2] - u[2] * n[1], u[2] * n[0] - u[0] * n[2], u[0] * n[1] - u[1] * n[0])
    def add_proj_views(self, offset_x=0, spacing=18):
        if self.dxf_doc is None:
            self.init_dxf_document()
        view_data = {}
        for name, cfg in VIEW_CFGS.items():
            visible, hidden, tan = self.cal_proj_hlr(cfg["N"], self.get_x_dir(cfg))
            all_items = visible + hidden + tan
            if not all_items:
                raise ValueError(f"{self.stp_path} {name} view no projection edge")
            bb = Compound.makeCompound(all_items).BoundingBox()
            view_data[name] = {"visible": visible, "hidden": hidden, "tan": tan, "bb": bb, "width": bb.xlen, "height": bb.ylen, "grid": cfg["grid"]}
        col_w = [
            view_data["left"]["width"],
            max(view_data["top"]["width"], view_data["front"]["width"], view_data["bottom"]["width"]),
            view_data["right"]["width"],
        ]
        row_h = [
            view_data["bottom"]["height"],
            max(view_data["left"]["height"], view_data["front"]["height"], view_data["right"]["height"]),
            view_data["top"]["height"],
        ]
        col_x = [offset_x, offset_x + col_w[0] + spacing, offset_x + col_w[0] + spacing + col_w[1] + spacing]
        row_y = [0, row_h[0] + spacing, row_h[0] + spacing + row_h[1] + spacing]
        count = 0
        for view in view_data.values():
            c, r = view["grid"]
            dx = float(col_x[c] + (col_w[c] - view["width"]) / 2 - view["bb"].xmin)
            dy = float(row_y[r] + (row_h[r] - view["height"]) / 2 - view["bb"].ymin)
            for items, layer in ((view["hidden"], "HID"), (view["visible"], "0"), (view["tan"], "TAN")):
                for item in items:
                    edge = item.translate((dx, dy, 0))
                    self.dxf_doc.add_shape(cq.Workplane("XY").newObject([edge]), layer)
                    count += 1
        return float(sum(col_w) + spacing * 2), count
    def save_dxf(self, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.saveas(save_path)
        return save_path

def main():
    hpr = STP_Tmp004()
    hpr.init_dxf_document()
    fnames, fpaths = get_files(STP_FOLDER, ["stp"])
    paths = sorted([Path(fpath) for fpath in fpaths], key=sort_key)
    offset_x = 0
    output_items = []
    for stp_path in paths:
        hpr.load_stp(stp_path)
        gather_data = hpr.gather_all()
        width, count = hpr.add_proj_views(offset_x)
        output_items.append({"stp_path": str(stp_path), "offset": offset_x, "width": width, "count": count, "plane": len(gather_data["plane"]), "cylinder": len(gather_data["cylinder"]), "cone": len(gather_data["cone"]), "torus": len(gather_data["torus"])})
        offset_x += width + 40
        print("ok", stp_path, output_items[-1])
    hpr.save_dxf(SAVE_PATH)
    print("ok", SAVE_PATH, len(hpr.doc.modelspace()))
if __name__ == "__main__":
    main()
