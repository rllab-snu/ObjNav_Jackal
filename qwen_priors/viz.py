"""Shared 12-sector heatmap renderer for object priors: draws, per topo node, a 12-wedge compass
(direction k -> screen angle (k*30+270)%360) over the occupancy map, colored turbo by score with the
value printed. Used by the final scoring pipeline (gen_direction, ensemble_priors)."""
import numpy as np
from PIL import Image, ImageDraw


def heatmap_rgb(score):
    import matplotlib.cm as cm
    r, g, b, _ = cm.turbo(float(max(0.0, min(1.0, score))))
    return (int(r * 255), int(g * 255), int(b * 255))


def render(target, scores, topo, occ, outpath, scale=6, R=58):
    ob = occ["ob"]; res = occ["res"]; ox = occ["ox"]; oz = occ["oz"]; H, W = ob.shape
    per_node = scores.get(target, {})              # object-first: {node: {dir: score}}
    bg = np.full((H, W, 3), 60, np.uint8); bg[occ["seen"] & ~ob] = 210; bg[ob] = 0
    img = Image.fromarray(bg).resize((W * scale, H * scale), Image.NEAREST)
    dr = ImageDraw.Draw(img)
    for node in topo.nodes:
        cc = (node.x - ox) / res * scale; cr = (node.z - oz) / res * scale
        for k in range(12):
            sc = per_node.get(str(node.id), {}).get(str(k), 0.0)
            a = (k * 30 + 270) % 360                       # dir yaw -> screen angle (0=+x, CW; up=270)
            dr.pieslice([cc - R, cr - R, cc + R, cr + R], a - 15, a + 15,
                        fill=heatmap_rgb(sc), outline=(255, 255, 255))
            mid = np.deg2rad(a); tx = cc + 0.62 * R * np.cos(mid); ty = cr + 0.62 * R * np.sin(mid)
            dr.text((tx - 12, ty - 5), f"{sc:.2f}", fill=(255, 255, 255))
        dr.ellipse([cc - 6, cr - 6, cc + 6, cr + 6], fill=(0, 0, 0), outline=(255, 255, 255))
        dr.text((cc + 9, cr - 13), f"n{node.id}", fill=(255, 0, 0))
    dr.text((10, 8), f"P(object='{target}' | direction)   turbo: 0=blue .. 1=red", fill=(255, 255, 255))
    img.save(outpath)
