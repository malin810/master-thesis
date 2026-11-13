"""SCRFD-34G: feste Eingabegröße (640 x 640) der ONNX-Datei aufheben.

Die Datei 34g.onnx (Hugging Face: RuteNL/SCRFD-face-detection-ONNX) wurde mit fester Größe exportiert.
Im Netz stehen deshalb feste Zielgrößen:
  * 6 x Reshape  ... [1, C, 80, 80] bzw. [1, C, 40, 40] bzw. [1, C, 20, 20]   (Gruppen-Normalisierung)
  * 2 x Resize   ... Zielgröße [1, 128, 40, 40] bzw. [1, 128, 80, 80]           (Hochskalieren, immer Faktor 2)

Dieses Skript ersetzt
  * die festen Reshape-Ziele durch die tatsächliche Größe des Tensors (Shape-Knoten),
  * die Resize-Zielgrößen durch "Faktor 2",
  * und macht Höhe und Breite der Eingabe variabel.
Rechnerisch ändert sich nichts: Bei 640 x 640 liefert die neue Datei dieselben Werte wie das Original
(das Skript prüft das und bricht sonst ab).

Aufruf (aus dem Projektordner; "onnx" wird nur dafür gebraucht und nicht ins Projekt installiert):
    uv run --isolated --no-project --with onnx --with onnxruntime --with numpy \
        python scripts/make_scrfd34g_dynamic.py models/scrfd_34g/34g.onnx models/scrfd_34g/34g_dynamic.onnx
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def make_dynamic(src, dst):
    model = onnx.load(src)
    g = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}

    # 1. Feste 4D-Ziele der Reshape-Knoten durch die echte Tensorgröße ersetzen
    new_nodes, n_reshape = [], 0
    for node in g.node:
        if node.op_type == "Reshape":
            target = inits.get(node.input[1])
            if target is not None and target.ndim == 1 and len(target) == 4 and (target > 0).all():
                instance_norm = producer[node.input[0]]
                assert instance_norm.op_type == "InstanceNormalization"
                first_reshape = producer[instance_norm.input[0]]
                assert first_reshape.op_type == "Reshape"
                original = first_reshape.input[0]            # Tensor vor der Gruppen-Normalisierung
                shape_name = f"dyn_shape_{n_reshape}"
                new_nodes.append(helper.make_node("Shape", [original], [shape_name], name="Shape_" + shape_name))
                node.input[1] = shape_name
                n_reshape += 1
        new_nodes.append(node)
    del g.node[:]
    g.node.extend(new_nodes)

    # 2. Resize: feste Zielgröße -> Faktor 2
    g.initializer.append(numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), name="scales_x2"))
    n_resize = 0
    for node in g.node:
        if node.op_type == "Resize":
            assert any(a.name == "mode" and a.s == b"nearest" for a in node.attribute)
            node.input[2] = "scales_x2"
            node.input[3] = ""
            n_resize += 1

    # 3. Eingabe und Ausgaben mit variabler Größe beschriften
    dims = g.input[0].type.tensor_type.shape.dim
    dims[2].dim_param, dims[3].dim_param = "H", "W"
    for out in g.output:
        for k, d in enumerate(out.type.tensor_type.shape.dim):
            if k == 0:
                d.dim_value = 1
            else:
                d.ClearField("dim_value")
                d.dim_param = f"{out.name}_d{k}"
    del g.value_info[:]
    onnx.save(model, dst)
    print(f"angepasst: {n_reshape} Reshape, {n_resize} Resize -> {dst}")


def check(src, dst):
    orig = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    new = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 640, 640)).astype(np.float32)
    a = orig.run(None, {orig.get_inputs()[0].name: x})
    b = new.run(None, {new.get_inputs()[0].name: x})
    diff = max(float(np.abs(p - q).max()) for p, q in zip(a, b))
    print(f"640 x 640: größte Abweichung zum Original = {diff:.2e}")
    assert diff < 1e-3, "Abweichung zu groß - Datei nicht verwenden"
    for w, h in [(1280, 736), (1920, 1088)]:
        outs = new.run(None, {new.get_inputs()[0].name: rng.standard_normal((1, 3, h, w)).astype(np.float32)})
        expected = [(w // s) * (h // s) * 2 for s in (8, 16, 32)]
        got = [o.shape[1] for o in outs[:3]]
        print(f"{w} x {h}: Anker pro Maßstab {got}, erwartet {expected}")
        assert got == expected


if __name__ == "__main__":
    make_dynamic(sys.argv[1], sys.argv[2])
    check(sys.argv[1], sys.argv[2])
