"""Shared render PANELS — one implementation used by BOTH habitat-sim (run_eval /
nav_viz_runner / exp_scripts) and the real robot (run_robot / robot.render).

Design
------
Each function renders ONE panel from read-only agent state (a localizer, an occupancy
map, an EarlyNav / ShieldLocalMap, a scan, a pose). No function knows the surrounding
LAYOUT — each caller arranges the panels it wants (see robot/render.py, nav_viz_runner).
Because the robot INHERITS the sim agent's localizer / controller / early_nav / shield
(same objects), passing those objects here yields byte-identical panels on sim and robot.

Ground truth (`gt` / `gt_position`) is an OPTIONAL argument: the sim passes it, the robot
passes None — same function either way.

Panel groups (see the module's public functions):
  cameras / detection : sam3_overlay          (rgb/depth panels are trivial from obs)
  localization        : ego_scan_panel, early_nav_occ_panel, shield_occ_panel,
                        likelihood_panel, particles_panel
  navigation          : topdown_nav_panel, dynmap_panel  (thin wrappers over
                        utils.nav_viz.render_topdown, whose behaviour is unchanged)
"""
import numpy as np

try:
    import cv2
except Exception:                       # pragma: no cover - viz optional in headless
    cv2 = None

from utils.nav_viz import render_topdown, _label, colorize_semantic

# ---- shared palette (RGB) -------------------------------------------------- #
GT_COLOR = (255, 255, 255)              # ground-truth pose (white)
EST_COLOR = (60, 120, 255)             # localizer estimate (blue)
GOAL_EXPLORE = (0, 220, 0)             # sub-goal while exploring (green)
GOAL_LASTMILE = (255, 60, 60)          # sub-goal in last-mile (red)
GOAL_SOURCES = (255, 210, 0)           # safe multi-source FMM terminal cells (yellow)
SUBGOAL_RAY = (0, 220, 220)            # early-nav frontier sub-goal heading (cyan)
DET_SELECTED = (0, 220, 0)             # detection: the chosen goal instance (green)
DET_CANDIDATE = (255, 60, 60)          # detection: other candidate instances (red)
SCAN_EST = (255, 120, 40)              # scan placed at the estimate (orange)
SCAN_GT = (90, 220, 120)               # scan placed at GT (green)


# ============================ low-level helpers ============================= #
def px(gmap, x, z, scale=1.0):
    """World (x, z) -> (col, row) in a `scale`-resized top-down image."""
    return (int(round((x - gmap.min_x) / gmap.res * scale)),
            int(round((z - gmap.min_z) / gmap.res * scale)))


def draw_pose(img, gmap, x, z, yaw, scale, color, r=5, hfov_deg=None, fan_range_m=2.5):
    """Filled dot + heading marker for a world pose on a top-down image (row=+z, col=+x).
    With hfov_deg the heading is drawn as a translucent FOV CONE (fan) of that angular width
    (matches the camera FOV); without it, a thin heading tick is drawn instead."""
    if cv2 is None:
        return
    cx, cy = px(gmap, x, z, scale)
    if not (0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]):
        return
    ang = np.arctan2(-np.cos(yaw), np.sin(yaw))          # image frame (forward = -z = up)
    if hfov_deg is not None:                             # FOV cone (fan)
        deg = np.degrees(ang)
        rad = max(6, int(round(fan_range_m / gmap.res * scale)))
        fan = img.copy()
        cv2.ellipse(fan, (cx, cy), (rad, rad), 0, deg - hfov_deg / 2, deg + hfov_deg / 2, color, -1)
        cv2.addWeighted(fan, 0.28, img, 0.72, 0, img)
        cv2.ellipse(img, (cx, cy), (rad, rad), 0, deg - hfov_deg / 2, deg + hfov_deg / 2,
                    color, 1, cv2.LINE_AA)
    else:                                                # thin heading tick (no FOV given)
        ex, ey = int(round(cx + 2.4 * r * np.cos(ang))), int(round(cy + 2.4 * r * np.sin(ang)))
        cv2.line(img, (cx, cy), (ex, ey), color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (20, 20, 20), 1, cv2.LINE_AA)


def base_map(gmap, out_h):
    """Plain top-down (navigable light / obstacle dark / unseen excluded). -> (img, scale)."""
    nav = getattr(gmap, "navigable", getattr(gmap, "free", None))
    img = np.full((gmap.H, gmap.W, 3), 90, np.uint8)
    if nav is not None:
        img[nav] = (205, 205, 205)
        img[~nav] = (45, 50, 65)
    seen = getattr(gmap, "seen", None)
    if seen is not None:
        img[~seen] = (70, 70, 80)
    scale = out_h / gmap.H
    if cv2 is not None:
        img = cv2.resize(img, (max(1, int(gmap.W * scale)), out_h),
                         interpolation=cv2.INTER_NEAREST)
    return img, scale


def _place_scan(scan, x, z, yaw):
    """Robot-frame scan (N,2 [x=right, z=back]) rotated by yaw + translated to world (x,z)."""
    ca, sa = np.cos(yaw), np.sin(yaw)
    wx = scan[:, 0] * ca - scan[:, 1] * sa + x
    wz = scan[:, 0] * sa + scan[:, 1] * ca + z
    return wx, wz


def _draw_world_points(img, gmap, wx, wz, scale, color):
    if cv2 is None or wx is None or not len(wx):
        return
    cols = np.clip(np.round((wx - gmap.min_x) / gmap.res * scale).astype(int), 0, img.shape[1] - 1)
    rows = np.clip(np.round((wz - gmap.min_z) / gmap.res * scale).astype(int), 0, img.shape[0] - 1)
    img[rows, cols] = color


# ================= ego (robot-centred) top-down primitive =================== #
def _ego_topdown(points, out_h=360, max_range=5.0, bg=32,
                 point_color=(90, 200, 255), point_r=2, hfov_deg=None):
    """Robot-centred top-down of robot-frame points (x=right, z=back, forward=-z=up).

    Shared by ego_scan_panel (single-frame scan) and shield_occ_panel (windowed buffer).
    With hfov_deg, a translucent camera-FOV cone (forward = up) is drawn under the points."""
    img = np.full((out_h, out_h, 3), bg, np.uint8)
    if cv2 is None:
        return img
    c = out_h // 2
    ppm = (out_h / 2 - 12) / max(1e-6, max_range)        # pixels per metre
    if hfov_deg is not None:                              # camera FOV cone (forward = up = -90 deg)
        rad = int(round(max_range * ppm))
        fan = img.copy()
        cv2.ellipse(fan, (c, c), (rad, rad), 0, -90 - hfov_deg / 2, -90 + hfov_deg / 2, (70, 130, 200), -1)
        cv2.addWeighted(fan, 0.22, img, 0.78, 0, img)
        cv2.ellipse(img, (c, c), (rad, rad), 0, -90 - hfov_deg / 2, -90 + hfov_deg / 2,
                    (95, 165, 235), 1, cv2.LINE_AA)
    for rr in range(1, int(max_range) + 1):              # 1 m range rings
        cv2.circle(img, (c, c), int(rr * ppm), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(img, (c, 0), (c, out_h), (55, 55, 55), 1)
    cv2.line(img, (0, c), (out_h, c), (55, 55, 55), 1)
    if points is not None and len(points):
        xs, zs = points[:, 0], points[:, 1]
        cols = np.round(c + xs * ppm).astype(int)
        rows = np.round(c + zs * ppm).astype(int)
        ok = (cols >= 0) & (cols < out_h) & (rows >= 0) & (rows < out_h)
        if point_r <= 1:
            img[rows[ok], cols[ok]] = point_color
        else:
            for pxp, pyp in zip(cols[ok], rows[ok]):
                cv2.circle(img, (int(pxp), int(pyp)), point_r, point_color, -1, cv2.LINE_AA)
    cv2.arrowedLine(img, (c, c), (c, c - int(0.8 * ppm)), (80, 255, 80), 2, tipLength=0.4)
    cv2.circle(img, (c, c), max(3, out_h // 90), (255, 255, 255), -1, cv2.LINE_AA)
    return img, ppm, c


# ============================ camera / detection =========================== #
def depth_panel(depth):
    """Habitat-format normalized [0,1] depth -> JET colormap RGB. Shared obs helper."""
    d = np.clip(np.asarray(depth).squeeze().astype(np.float32), 0.0, 1.0)
    u8 = (d * 255).astype(np.uint8)
    if cv2 is None:
        return np.stack([u8] * 3, -1)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def sam3_overlay(rgb, mask, color=(255, 40, 40)):
    """Detection washed onto an RGB frame (fade rgb toward white, paint the detection
    bright) -- the shared 'make the detection/SAM 3 visualization image' step.

    `mask` may be either a plain boolean HxW mask (painted in `color`), OR a per-instance
    dict {"masks": [HxW bool, ...], "selected": idx|None}: then every candidate instance is
    painted red and the SELECTED goal instance green, so the picture shows which detection
    last-mile is homing on. Env-agnostic (sim SAM 3 / robot SAM 3)."""
    rgbf = np.asarray(rgb)[..., :3].astype(np.float32)
    out = (0.35 * rgbf + 0.65 * 255).astype(np.uint8)

    def _paint(m, col):
        m = np.asarray(m)
        if cv2 is not None and m.shape[:2] != out.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (out.shape[1], out.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        m = m.astype(bool)
        if m.any():
            out[m] = (0.15 * rgbf[m] + 0.85 * np.array(col)).astype(np.uint8)

    if isinstance(mask, dict):                       # per-instance: candidates red, selected green
        masks = mask.get("masks") or []
        sel = mask.get("selected")
        for i, m in enumerate(masks):
            if i != sel:
                _paint(m, DET_CANDIDATE)
        if sel is not None and 0 <= sel < len(masks):
            _paint(masks[sel], DET_SELECTED)         # painted last -> wins any overlap
    elif mask is not None:
        _paint(mask, color)
    return out


# ============================== localization =============================== #
def ego_scan_panel(scan, out_h=360, max_range=5.0, hfov_deg=None):
    """The current single depth frame as an egocentric 2-D scan (robot at centre, forward
    up), with the camera FOV cone when hfov_deg is given. This is exactly what the localizer
    matches against the map."""
    img = _ego_topdown(scan, out_h, max_range, point_color=(90, 200, 255), point_r=2,
                       hfov_deg=hfov_deg)
    return img[0] if isinstance(img, tuple) else img


def shield_occ_panel(shield, out_h=360, max_range=5.0):
    """The post-convergence collision-shield's SHORT-WINDOW local occupancy map
    (`ShieldLocalMap`): the last ~`window` frames' obstacle points, robot-centred, plus the
    swept forward BODY footprint the shield tests. Odometry-framed (correct even when the
    GLOBAL localizer is mis-converged)."""
    recent = list(getattr(shield, "_recent", []) or [])
    pts_robot = None
    if recent:
        P = np.concatenate(recent, axis=0)               # odo-frame points (M,2)
        ca, sa = np.cos(shield.oyaw), np.sin(shield.oyaw)
        dx = P[:, 0] - shield.ox; dz = P[:, 1] - shield.oz
        rx = dx * ca + dz * sa                            # robot frame: x = right
        rz = -dx * sa + dz * ca                           # forward = -z
        pts_robot = np.stack([rx, rz], axis=1)
    img, ppm, c = _ego_topdown(pts_robot, out_h, max_range,
                               point_color=(255, 150, 60), point_r=2)
    if cv2 is not None:                                   # swept footprint (one step, body-width)
        rad, fwd = shield.rad, shield.fwd
        x0 = int(c - (rad + 0.10) * ppm); x1 = int(c + (rad + 0.10) * ppm)
        y0 = int(c - rad * ppm); y1 = int(c + (fwd + rad + 0.15) * ppm)
        cv2.rectangle(img, (x0, y0), (x1, y1), (70, 120, 200), 1, cv2.LINE_AA)
    return img


def early_nav_occ_panel(en, out_h=360, show_combo=True):
    """EarlyNav's PRE-convergence full-accumulation local occupancy map (odometry-framed),
    with the combo policy state: the persistent frontier SUB-GOAL heading (cyan ray) and, if
    the goal was detected pre-convergence, its odo-anchored position (red marker). Reset on
    re-globalize is reflected automatically (the grids/pose are wiped in en.reset_map())."""
    H, W = en.H, en.W
    img = np.full((H, W, 3), 60, np.uint8)               # unknown (mid-dark)
    free = (en.free > 0) & (en.occ == 0)
    img[free] = (150, 150, 150)
    img[en.occ >= 1] = (40, 45, 60)                      # obstacle (dark)
    scale = out_h / H
    if cv2 is None:
        return img
    rr = int(round(en.oz / en.res) + en.c0)              # robot cell (row=+z, col=+x)
    rc = int(round(en.ox / en.res) + en.c0)
    L = max(8, H // 12)

    def ray(yaw, color, length, width=1):
        ex = int(round(rc + length * np.sin(yaw)))       # forward=(sin yaw,-cos yaw) in (x,z)
        ez = int(round(rr - length * np.cos(yaw)))
        cv2.line(img, (rc, rr), (ex, ez), color, width, cv2.LINE_AA)

    vc = getattr(en, "_viz_combo", None) if show_combo else None
    if show_combo:
        # combo policy FAN: each discrete candidate (forward / turn-left / turn-right) drawn
        # as a ray in its resulting heading, length by its combo score, the CHOSEN one green
        # and an unsafe forward a dim-red stub. (en._viz_combo is stored by _combo_action --
        # viz-only, no control effect.)
        if vc and vc.get("cands"):
            scored = [s for _, s in vc["cands"] if s is not None]
            smax = max(scored) if scored else 1.0
            for dh, s in vc["cands"]:
                ya = en.oyaw + dh
                if s is None:
                    ray(ya, (120, 60, 60), int(0.5 * L))              # unsafe forward
                else:
                    chosen = vc.get("best_dh") is not None and abs(dh - vc["best_dh"]) < 1e-6
                    ln = int(L * (0.5 + 0.9 * (s / max(smax, 1e-6))))
                    ray(ya, (60, 230, 60) if chosen else (230, 180, 60), ln, 3 if chosen else 1)
        # persistent frontier SUB-GOAL heading (cyan) + pre-conv DETECTED goal (red cross)
        if getattr(en, "_subgoal_yaw", None) is not None:
            ray(en._subgoal_yaw, SUBGOAL_RAY, int(1.7 * L))
        if getattr(en, "goal_odo", None) is not None:
            gr = int(round(en.goal_odo[1] / en.res) + en.c0)
            gc = int(round(en.goal_odo[0] / en.res) + en.c0)
            if 0 <= gr < H and 0 <= gc < W:
                cv2.drawMarker(img, (gc, gr), GOAL_LASTMILE, cv2.MARKER_TILTED_CROSS,
                               max(6, H // 40), 2)
    if 0 <= rr < H and 0 <= rc < W:
        if not (vc and vc.get("cands")):
            ray(en.oyaw, (80, 255, 80), L)               # heading tick when no combo fan
        cv2.circle(img, (rc, rr), max(3, H // 60), (255, 255, 255), -1, cv2.LINE_AA)
    return cv2.resize(img, (max(1, int(W * scale)), out_h), interpolation=cv2.INTER_NEAREST)


def likelihood_panel(loc, gmap, scan, out_h=360, gt=None, est=None, hfov_deg=None):
    """Best-over-yaw single-frame observation likelihood of every explorable cell -- "where
    on the map does THIS one frame fit" (viridis). Comes straight from the localizer's own
    scorer (loc.score_volume): full scan, the localizer's heading resolution, the SAME 3-term
    score (endpoint + free + cone-unseen) that weights the particles."""
    img, scale = base_map(gmap, out_h)
    if cv2 is not None and scan is not None and len(scan) >= 10:
        _, vol = loc.score_volume(scan)                  # (nb, H, W)
        best = vol.max(axis=0)
        expl = getattr(gmap, "explorable", gmap.free)
        vals = best[expl]
        lo, hi = np.percentile(vals, 2), vals.max()
        heat = np.zeros((gmap.H, gmap.W), np.float32)
        heat[expl] = np.clip((best[expl] - lo) / max(1e-6, hi - lo), 0, 1)
        heat = cv2.resize(heat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        cmap = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype(np.uint8),
                                              cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        mask = cv2.resize(expl.astype(np.float32), (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_LINEAR) > 0.5
        img[mask] = (0.25 * img[mask] + 0.75 * cmap[mask]).astype(np.uint8)
    r = max(4, out_h // 90)
    if gt is not None:
        draw_pose(img, gmap, gt[0], gt[1], gt[2], scale, GT_COLOR, r, hfov_deg=hfov_deg)
    if est is not None:
        draw_pose(img, gmap, est[0], est[1], est[2], scale, EST_COLOR, r, hfov_deg=hfov_deg)
    return img


def particles_panel(loc, gmap, out_h=360, gt=None, est=None, scan=None, mode="heatmap", hfov_deg=None):
    """The particle filter's FINAL state: the particle cloud + the weighted estimate, plus
    (sim only, when supplied) the GT pose. `mode`: 'heatmap' = weight-density (autumn, the
    offline/analysis view), 'dots' = raw particle positions (fast live view). If `scan` is
    given it is drawn placed at the estimate (orange) and at the GT (green) -- different
    colours -- to show how the single frame lines up under each pose."""
    img, scale = base_map(gmap, out_h)
    if cv2 is None:
        return img
    P = getattr(loc, "particles", None)
    if P is not None and len(P):
        if mode == "dots":
            _draw_world_points(img, gmap, P[:, 0], P[:, 1], scale, (0, 200, 255))
        else:
            w = getattr(loc, "weights", np.ones(len(P)) / len(P))
            cols = np.clip(((P[:, 0] - gmap.min_x) / gmap.res).astype(int), 0, gmap.W - 1)
            rows = np.clip(((P[:, 1] - gmap.min_z) / gmap.res).astype(int), 0, gmap.H - 1)
            acc = np.zeros((gmap.H, gmap.W), np.float64)
            np.add.at(acc, (rows, cols), w)
            if acc.max() > 0:
                d = cv2.dilate((acc / acc.max()).astype(np.float32), np.ones((5, 5), np.uint8))
                d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                cmap = cv2.cvtColor(cv2.applyColorMap((255 - d * 255).astype(np.uint8),
                                                      cv2.COLORMAP_AUTUMN), cv2.COLOR_BGR2RGB)
                m = d > 0.02
                img[m] = (0.3 * img[m] + 0.7 * cmap[m]).astype(np.uint8)
    if scan is not None and len(scan):
        if est is not None:
            _draw_world_points(img, gmap, *_place_scan(scan, est[0], est[1], est[2]), scale, SCAN_EST)
        if gt is not None:
            _draw_world_points(img, gmap, *_place_scan(scan, gt[0], gt[1], gt[2]), scale, SCAN_GT)
    r = max(4, out_h // 90)
    if gt is not None:
        draw_pose(img, gmap, gt[0], gt[1], gt[2], scale, GT_COLOR, r, hfov_deg=hfov_deg)
    if est is not None:
        draw_pose(img, gmap, est[0], est[1], est[2], scale, EST_COLOR, r, hfov_deg=hfov_deg)
    return img


# =============================== navigation ================================ #
def topdown_nav_panel(gmap, est_pos, est_heading, subgoal=None, last_mile=False,
                      gt_position=None, fmm_dist=None, hfov_deg=None, out_height=480,
                      stg_position=None, object_target=None, goal_sources=None):
    """Global top-down: occupancy + FMM field + skeleton + agent fan.

    Explore draws its subgoal green. Last-mile draws the raw detected object red,
    every safe multi-source terminal cell yellow, and the active STG cyan. Sim-only
    ground truth remains a blue X.
    """
    display_goal = object_target if last_mile and object_target is not None else subgoal
    return render_topdown(
        gmap, est_pos, est_heading, goal=display_goal, fmm_dist=fmm_dist,
        show_fmm=True, show_skeleton=True, hfov_deg=hfov_deg, out_height=out_height,
        goal_color=(GOAL_LASTMILE if last_mile else GOAL_EXPLORE),
        gt_position=gt_position, stg_position=stg_position,
        goal_sources=goal_sources, goal_source_color=GOAL_SOURCES)


def dynmap_panel(gmap, est_pos, est_heading, subgoal=None, gt_position=None,
                 hfov_deg=None, out_height=480, last_mile=False,
                 object_target=None):
    """Online dynamic-map view: base occupancy + NEW obstacles the live scans added (orange),
    NO FMM/skeleton so the map update is isolated. Same render_topdown, show_fmm/skeleton off
    (its dyn_occ overlay is what draws the new obstacles)."""
    display_goal = object_target if last_mile and object_target is not None else subgoal
    return render_topdown(
        gmap, est_pos, est_heading, goal=display_goal, fmm_dist=None,
        show_fmm=False, show_skeleton=False, hfov_deg=hfov_deg, out_height=out_height,
        goal_color=(GOAL_LASTMILE if last_mile else GOAL_EXPLORE),
        gt_position=gt_position)


# ===================== composition (rows / grid stack) ===================== #
def hrow(panels, labels, height, sep=8):
    """Resize panels to a common `height`, add a label STRIP ABOVE each (never covering
    content), hconcat with separators -> one row. Every panel gets the strip (blank when no
    label) so heights stay uniform; the separators match that label-inclusive height."""
    out = []
    for p, lb in zip(panels, labels):
        p = np.ascontiguousarray(np.asarray(p)[..., :3]).astype(np.uint8)
        if cv2 is not None and p.shape[0] != height:
            p = cv2.resize(p, (max(1, int(p.shape[1] * height / p.shape[0])), height))
        out.append(_label(p, lb))                       # label strip above (uniform +band)
    if not out:
        return np.zeros((height, 1, 3), np.uint8)
    row_h = out[0].shape[0]                              # height + label band
    bar = np.full((row_h, sep, 3), (20, 20, 20), np.uint8)
    pieces = []
    for i, pan in enumerate(out):
        if i:
            pieces.append(bar)
        pieces.append(pan)
    return np.concatenate(pieces, axis=1)


def _text_bar(width, lines, thick=1, max_scale=0.55, ref_len=100):
    """Render status lines as a MONOSPACE grid -- every glyph in a fixed-width cell -- so the
    '|' partitions (and every fixed-width field) sit at FIXED x-positions and never shift as
    the values change. The cell width + font scale + line height are derived from `width` and
    a CONSTANT `ref_len` (assumed max line length), NOT the per-frame text, so every frame is
    the same size (required for the mp4 writer) and the columns stay put across frames.
    Partition bars are tinted so the columns read clearly."""
    if cv2 is None or not lines:
        return np.zeros((1, max(1, width), 3), np.uint8)
    cell = max(6.0, (width - 16) / ref_len)                 # constant per width -> stable
    (cw1, _), _ = cv2.getTextSize("M", cv2.FONT_HERSHEY_DUPLEX, 1.0, thick)
    scale = min(max_scale, max(0.30, (cell - 2) / max(1, cw1)))   # glyph fits the cell
    (_, ch), base = cv2.getTextSize("M", cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    lh = ch + base + 12
    bar = np.zeros((int(lh * len(lines) + 6), width, 3), np.uint8)
    for li, line in enumerate(lines):
        y = int(lh * (li + 1) - 8)
        for ci, g in enumerate(line):
            if g == " ":
                continue
            x = int(8 + ci * cell)
            color = (120, 170, 255) if g == "|" else (240, 240, 240)
            cv2.putText(bar, g, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)
    return bar


def vstack(rows, text_lines=None):
    """Pad rows to a common width and vconcat, with an optional (monospace) text bar on top."""
    rows = [r for r in rows if r is not None]
    if not rows:
        return np.zeros((1, 1, 3), np.uint8)
    w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < w:
            r = np.concatenate([r, np.full((r.shape[0], w - r.shape[1], 3), (20, 20, 20), np.uint8)], axis=1)
        padded.append(r)
    grid = np.concatenate(padded, axis=0)
    if text_lines and cv2 is not None:
        grid = np.concatenate([_text_bar(w, text_lines), grid], axis=0)
    return grid


def _geo_dist(ctrl, est):
    """Geodesic distance (m) from the estimate cell to the controller's FMM target, along the
    free-space geodesic field (not straight line). Available on sim AND robot (no GT needed)."""
    if ctrl is None or est is None:
        return None
    fd = getattr(ctrl, "fmm_dist", None)
    gmap = getattr(ctrl, "map", None)
    if fd is None or gmap is None:
        return None
    try:
        rc = gmap.point_to_grid(np.array([est[0], 0.0, est[1]]))
        r = int(np.clip(round(rc[0]), 0, gmap.H - 1))
        c = int(np.clip(round(rc[1]), 0, gmap.W - 1))
        v = float(fd[r, c]) * gmap.res
        return v if np.isfinite(v) else None
    except Exception:
        return None


def navigation_stage(agent):
    """Short stage name shared by rendered status and structured robot logs."""
    a = agent
    loc = getattr(a, "localizer", None)
    en = getattr(a, "early_nav", None)
    conv = getattr(loc, "converged", None) if loc is not None else None
    # Short shared stage names for both sim and robot renderers. Pre-convergence
    # EarlyNav owns control first; its local last-mile commit is shown separately.
    mode = getattr(a, "mode", "?")
    loc_steps = getattr(a, "_loc_steps", 0)
    loc_budget = getattr(a, "max_localize_steps", None)
    within_loc_budget = loc_budget is None or loc_steps < loc_budget
    early_active = (loc is not None and conv is False and en is not None
                    and getattr(a, "use_early_nav", True) and within_loc_budget)
    if early_active:
        stage = "lastmile_early" if getattr(en, "_committed", False) else "early"
    elif mode == "last_mile":
        stage = "lastmile"
    elif mode == "explore" and getattr(a, "_topo_phase", None) == "prior_hint":
        stage = "prior"
    elif mode == "explore" and (
            getattr(a, "_topo_phase", None) == "score_search"
            or getattr(a, "search_mode", None) == "topo"):
        stage = "score"
    else:
        stage = mode
    return stage


def status_lines(agent, step=None, action=None, est=None, gt=None, extra=None):
    """Overlay text lines, shared by the sim + robot renderers so both show the same info.
    Fields on each line are separated by '  |  '. est / gt = (x, z, yaw[rad]); the sim passes
    gt (the true agent pose), the robot passes None. `extra` carries the GT-ONLY sim metrics
    (dist_to_goal, d_obj) -- everything else is derived from the agent's own state so it shows
    on the robot too. Lines: [1] step/goal/stage/action  [2] localization  [3] pose(+GT,ATE)
    [4] navigation distances."""
    a = agent
    extra = extra or {}
    row = lambda segs: " | ".join(s for s in segs if s)
    loc = getattr(a, "localizer", None)
    en = getattr(a, "early_nav", None)
    conv = getattr(loc, "converged", None) if loc is not None else None
    stage = navigation_stage(a)
    cat = getattr(a, "goal_category", "?")
    prompt = getattr(a, "sam3_prompt", None)
    goal_txt = f"'{prompt}'" if prompt and prompt != cat else f"'{cat}'"

    shield_txt = None
    shield_raw = getattr(a, "_shield_raw_action", None)
    shield_action = getattr(a, "_shield_action", None)
    actions = getattr(getattr(a, "controller", None), "A", {})
    if shield_raw is not None and shield_action is not None and shield_raw != shield_action:
        action_names = {value: str(name).replace("_", " ") for name, value in actions.items()}
        raw_name = action_names.get(shield_raw, str(shield_raw))
        shield_name = action_names.get(shield_action, str(shield_action))
        shield_txt = f"shield={raw_name}->{shield_name}"

    lines = [row([f"step {('?' if step is None else step):>4}",
                  f"goal {goal_txt}".ljust(24),
                  f"stage={stage}".ljust(15),
                  (f"action={action}".ljust(17) if action is not None else None),
                  shield_txt])]

    # [2] localization: converged / settled (stricter, sustained) / pre-conv last-mile commit /
    #     particle spread. (oracle pose has no localizer.)  Bools ljust to a fixed width.
    if loc is not None:
        seg = [f"converged={str(conv):<5}"]
        if getattr(loc, "settled", None) is not None:
            seg.append(f"settled={str(loc.settled):<5}")
        if en is not None and getattr(en, "_committed", None) is not None:
            seg.append(f"lastmile_committed={str(en._committed):<5}")
        if getattr(loc, "spread", None) is not None:
            seg.append(f"spread={loc.spread:5.2f}m")
        lines.append("loc:  " + row(seg))
    else:
        lines.append("loc:  oracle pose")

    # [3] estimate pose (fixed-width x/z coords + yaw deg); GT + ATE only when supplied (sim).
    def _p(p):                       # cv2 Hershey is ASCII-only -> "deg", not the ° glyph
        return f"({p[0]:+6.2f},{p[1]:+6.2f},{np.degrees(p[2]):4.0f}deg)"
    if est is not None:
        seg = [f"est (x,z,yaw)={_p(est)}"]
        if gt is not None:
            seg.append(f"gt={_p(gt)}")
            seg.append(f"ATE={float(np.hypot(est[0]-gt[0], est[1]-gt[1])):5.2f}m")
        lines.append(row(seg))

    # [4] navigation distances: FMM status + geodesic dist + d(subgoal) are agent-derived
    #     (sim AND robot); d2goal / d(obj) need GROUND TRUTH -> sim only, via `extra`.
    seg = []
    ctrl = getattr(a, "controller", None)
    if ctrl is not None and getattr(ctrl, "last_status", None) is not None:
        seg.append(f"fmm={str(ctrl.last_status):<7}")
    geo = _geo_dist(ctrl, est)
    if geo is not None:
        seg.append(f"geo={geo:6.2f}m")
    sg = getattr(a, "subgoal", None)
    en = getattr(a, "early_nav", None)
    loc = getattr(a, "localizer", None)
    # Pre-convergence early_nav homes in its own ODOMETRY frame (the global pose is unknown),
    # so the world-frame est<->subgoal distance is meaningless -- and self.subgoal is still the
    # stale explore target. While early_nav is committed to the local last-mile, show the
    # odo-frame distance to its ray-navigable subgoal: exactly what STOP compares to stop_distance.
    en_tgt = (getattr(en, "_lm_subgoal_odo", None) or getattr(en, "goal_odo", None)) if en else None
    if (en is not None and en_tgt is not None and getattr(en, "_committed", False)
            and loc is not None and not getattr(loc, "converged", True)):
        d = float(np.hypot(en_tgt[0] - en.ox, en_tgt[1] - en.oz))
        seg.append(f"d(subgoal,odo)={d:6.2f}m")
    elif sg is not None and est is not None:
        seg.append(f"d(subgoal)={float(np.hypot(est[0]-sg[0], est[1]-sg[2])):6.2f}m")
    if extra.get("dist_to_goal") is not None:
        seg.append(f"d2goal(GT)={extra['dist_to_goal']:6.2f}m")
    if extra.get("d_obj") is not None:
        seg.append(f"d(obj,GT)={extra['d_obj']:6.2f}m")
    if seg:
        lines.append(row(seg))
    return lines


# ===================== theme GROUP builders (obs / loc / nav) =============== #
# Each returns row image(s) built from the panels above, so BOTH the sim (nav_viz_runner)
# and the robot (RobotRenderer) get the SAME rendering per group; each caller only decides
# the overall stacking / text bar / live-window vs video.

def _local_occ(agent, out_h):
    """Pick the real-time local occupancy map: EarlyNav (pre-convergence, combo state) while
    it is in use and unconverged, else the ShieldLocalMap short window. -> (panel, label)."""
    loc = getattr(agent, "localizer", None)
    en = getattr(agent, "early_nav", None)
    sh = getattr(agent, "_shield_map", None)
    converged = bool(getattr(loc, "converged", False)) if loc is not None else True
    if (en is not None and getattr(agent, "use_early_nav", True) and not converged
            and (en.occ.any() or en.free.any())):
        return early_nav_occ_panel(en, out_h), "local map: early-nav + combo"
    if sh is not None and (sh._recent if hasattr(sh, "_recent") else None):
        return shield_occ_panel(sh, out_h), "local map: shield window"
    if en is not None:
        return early_nav_occ_panel(en, out_h), "local map: early-nav"
    return None, None


def obs_detection_rows(obs, out_h, sam3_masks=None, want_semantic=True, prompt=None):
    """OBS + DETECTION group: main (rgb|depth|[semantic]|[sam3]) and, if present, sub
    (rgb_top|depth_top|[semantic_top]|[sam3]). `sam3_masks` = {'main':mask,'sub':mask} from
    the caller (robot: SAM 3 client; sim: detector masks or None). -> list of row images."""
    sam3_masks = sam3_masks or {}
    rows = []
    for rgb_k, dep_k, sem_k, tag, mk in [("rgb", "depth", "semantic", "main", "main"),
                                         ("rgb_top", "depth_top", "semantic_top", "sub", "sub")]:
        if rgb_k not in obs:
            continue
        rgb = np.asarray(obs[rgb_k])[..., :3].astype(np.uint8)
        p, l = [rgb, depth_panel(obs[dep_k])], [f"{tag} rgb", f"{tag} depth"]
        if want_semantic and sem_k in obs:
            p.append(colorize_semantic(obs[sem_k])); l.append(f"{tag} semantic")
        if mk in sam3_masks:
            p.append(sam3_overlay(rgb, sam3_masks[mk]))
            l.append(f"{tag} sam3" + (f" '{prompt}'" if prompt else ""))
        rows.append(hrow(p, l, out_h))
    return rows


def localization_row(agent, obs, out_h, est, gt=None, hfov_deg=None):
    """LOCALIZATION group (one row): ego-scan | real-time local occupancy map | single-frame
    likelihood | particle-filter state (particles + estimate, + GT pose when supplied). `est`
    / `gt` are (x, z, yaw); the sim passes both, the robot passes gt=None. `hfov_deg` draws the
    camera FOV cone on the ego-scan and the pose markers. None if no localizer."""
    loc = getattr(agent, "localizer", None)
    if loc is None:
        return None
    scan = loc.extract_scan(obs, "depth")
    gmap = agent.controller.map
    p, l = [ego_scan_panel(scan, out_h, hfov_deg=hfov_deg)], ["ego scan"]
    lo, lolab = _local_occ(agent, out_h)
    if lo is not None:
        p.append(lo); l.append(lolab)
    p.append(likelihood_panel(loc, gmap, scan, out_h, gt=gt, est=est, hfov_deg=hfov_deg))
    l.append("single-frame likelihood")
    p.append(particles_panel(loc, gmap, out_h, gt=gt, est=est, scan=scan, hfov_deg=hfov_deg))
    l.append("particles + est" + (" + GT" if gt is not None else ""))
    return hrow(p, l, out_h)


VISITED_RGB = (70, 70, 70)              # visited views -> solid dark grey
PRIOR_HINT_RGB = (255, 210, 0)          # current prior-hint view -> yellow outline


def _score_rgb(s):
    """Scalar 0-1 -> turbo RGB, remapped to skip turbo's near-black low end so low reads as a
    moderately dark BLUE (not near-black purple); high stays red. TURBO if available, else JET."""
    s = max(0.0, min(1.0, float(s)))
    t = 0.15 + 0.85 * s                 # 0 -> turbo(0.15) medium-dark blue; 1 -> turbo(1) red
    cmap = getattr(cv2, "COLORMAP_TURBO", getattr(cv2, "COLORMAP_JET", 2))
    bgr = cv2.applyColorMap(np.uint8([[int(round(t * 255))]]), cmap)[0, 0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def topo_panel(agent, out_h=360):
    """TOPO prior + visited-graph over the occupancy map (only in --lastmile_search topo): per node,
    a 12-wedge turbo compass of the goal's per-view prior (dir_00 = up, +yaw CW); VISITED views are
    dimmed, and the CURRENT view is outlined yellow during prior hint or white during score search.
    Read-only from agent state."""
    gmap = agent.controller.map
    img, scale = base_map(gmap, out_h)
    topo = getattr(agent, "_topo_cached", None)
    if cv2 is None or topo is None:
        return img
    prior = getattr(agent, "_prior", None) or {}
    visited = getattr(agent, "_visited", None) or {}
    cn, fd = getattr(agent, "_cur_node", None), getattr(agent, "_face_dir", None)
    R = max(9, int(round(0.7 / gmap.res * scale)))          # ~0.7 m compass radius
    for node in topo.nodes:
        cx, cy = px(gmap, node.x, node.z, scale)
        per = prior.get(str(node.id), {}); vgn = visited.get(str(node.id), {})
        for k in range(12):
            col = VISITED_RGB if vgn.get(str(k)) else _score_rgb(per.get(str(k), 0.0))
            a0 = (k * 30 + 270 - 15) % 360                  # dir_00 = up (270), +yaw CW
            cv2.ellipse(img, (cx, cy), (R, R), 0, a0, a0 + 30, col, -1)
        # black view boundaries: outer ring + 12 radial spokes
        cv2.circle(img, (cx, cy), R, (0, 0, 0), 1, cv2.LINE_AA)
        for k in range(12):
            ang = np.deg2rad((k * 30 + 270 - 15) % 360)
            ex, ey = int(round(cx + R * np.cos(ang))), int(round(cy + R * np.sin(ang)))
            cv2.line(img, (cx, cy), (ex, ey), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(img, f"n{node.id}", (cx + R + 2, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 0), 1, cv2.LINE_AA)
    if cn is not None and fd is not None:                   # highlight current subgoal view
        n = topo.node(int(cn))
        if n is not None:
            cx, cy = px(gmap, n.x, n.z, scale)
            a0 = (int(fd) * 30 + 270 - 15) % 360
            is_hint = getattr(agent, "_topo_phase", None) == "prior_hint"
            color = PRIOR_HINT_RGB if is_hint else (255, 255, 255)
            cv2.ellipse(img, (cx, cy), (R, R), 0, a0, a0 + 30,
                        color, 3 if is_hint else 2, cv2.LINE_AA)
    _score_legend(img)
    return img


def _score_legend(img, w=110, h=9):
    """Small turbo colorbar (low->high) at the panel's bottom-left, so the spectrum is legible."""
    H, W = img.shape[:2]
    x0, y0 = 8, H - h - 16
    for i in range(w):
        img[y0:y0 + h, x0 + i] = _score_rgb(i / (w - 1))
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), 1)
    cv2.putText(img, "P(obj) low", (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(img, "high", (x0 + w - 30, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0 + h + 4), (x0 + 12, y0 + h + 14), VISITED_RGB, -1)
    cv2.rectangle(img, (x0, y0 + h + 4), (x0 + 12, y0 + h + 14), (0, 0, 0), 1)
    cv2.putText(img, "= visited", (x0 + 16, y0 + h + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 235, 235), 1, cv2.LINE_AA)


def navigation_row(agent, out_h, est_pos, est_heading, gt_position=None, hfov_deg=None):
    """NAVIGATION group (one row): top-down FMM (occ + geodesic field + skeleton + sub-goal +
    estimate/GT + current STG) | dynamic map (base + online NEW obstacles) | topo
    prior+visited (topo search only)."""
    gmap = agent.controller.map
    subgoal = getattr(agent, "subgoal", None)
    last_mile = getattr(agent, "mode", None) == "last_mile"
    controller = getattr(agent, "controller", None)
    fmm_dist = getattr(controller, "fmm_dist", None)
    stg_position = getattr(controller, "last_stg_world", None)
    object_target = getattr(agent, "last_mile_goal", None) if last_mile else None
    goal_sources = (getattr(agent, "_last_mile_source_mask", None)
                    if last_mile else None)
    td = topdown_nav_panel(
        gmap, est_pos, est_heading, subgoal=subgoal, last_mile=last_mile,
        gt_position=gt_position, fmm_dist=fmm_dist, hfov_deg=hfov_deg,
        out_height=out_h, stg_position=stg_position,
        object_target=object_target, goal_sources=goal_sources)
    dyn = getattr(gmap, "dyn_occ", None)
    n_dyn = int(dyn.sum()) if dyn is not None else 0
    dm = dynmap_panel(
        gmap, est_pos, est_heading, subgoal, gt_position, hfov_deg, out_h,
        last_mile=last_mile, object_target=object_target)
    panels, labels = [td, dm], ["top-down: FMM", f"top-down: dynamic map (+{n_dyn} new)"]
    if getattr(agent, "search_mode", None) == "topo" and getattr(agent, "_topo_cached", None) is not None:
        panels.append(topo_panel(agent, out_h))
        labels.append(f"topo prior+visited (goal '{getattr(agent, 'goal_category', '')}')")
    return hrow(panels, labels, out_h)
