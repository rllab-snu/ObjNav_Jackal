"""Observation-based localization on a KNOWN occupancy map (no oracle pose).

Replaces ``sim.get_agent_state()`` with a pose estimated from what the agent can
actually observe: the executed action (motion model), the bump flag, and the
depth camera (measurement model) matched against the pre-built global occupancy
map.

Three shared ingredients, two estimators:

  * **motion model** -- dead reckoning from the discrete actions
    (forward = ``forward_step_size``, turn = ``turn_angle``). A commanded forward
    that ends in a collision moves the agent by ~0, so the bump flag
    (``sim.previous_step_collided``) zeroes it out; without this the estimate
    drifts badly since ``allow_sliding=False``.
  * **scan** -- the depth image back-projected and sliced to the same height band
    the map was built from, flattened to 2D points in the robot frame. A single
    narrow-FOV frame is often ambiguous (sliding along a corridor is unobservable),
    so the last K frames are ACCUMULATED: buffered scans are pushed through the
    inverse motion at every step, so they stay expressed in the *current* robot
    frame and merge into a wide-FOV local map.
  * **likelihood field** -- ``distanceTransform`` of the map gives, per cell, the
    distance to the nearest obstacle. A scan point landing d metres from an
    obstacle scores ``exp(-d^2 / 2 sigma^2)``. It is bounded (an unmapped object
    contributes ~0 instead of dragging the fit, unlike a least-squares/ICP cost)
    and is a table lookup, so scoring a pose is O(#points).

Main estimator (this module):
  * ``ParticleFilterLocalizer`` -- global Monte-Carlo localization with
    SINGLE-FRAME scan measurements and cumulative weights. Dense particles over the
    explorable map collapse onto the true pose as motion brings in new evidence.

Retired variants were moved out of the package to ``legacy/localizers.py`` (git-
untracked, not imported here): ``ScanMatchLocalizer`` and ``DeadReckonLocalizer``
assume a known start; ``AccumGridParticleFilter`` is the older accumulate-scans +
grid-sweep filter. Future work: extend the main filter with an accumulated local map.
"""
from collections import deque

import numpy as np
import cv2

from .geometry import backproject_pixels, vertical_half_fov_down


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot2d(pts, a):
    """Rotate (N,2) points [x, z] by angle a (same convention as the world yaw)."""
    ca, sa = np.cos(a), np.sin(a)
    x, z = pts[:, 0], pts[:, 1]
    return np.stack([x * ca - z * sa, x * sa + z * ca], axis=-1)


class LikelihoodField:
    """distance-to-nearest-obstacle table + a bounded scoring function."""

    def __init__(self, gmap, sigma_m=0.15, trim=0.8, d_max=1.0, n_eff=100.0,
                 miss_mult=8.0):
        self.g = gmap
        self.sigma = float(sigma_m)
        self.trim = float(trim)          # keep this best fraction of points
        self.d_max = float(d_max)        # per-point distance cap (robust kernel)
        # Off-map / unseen points are a HARD miss (no evidence at all), penalised
        # far more than a merely-far in-map point: miss_cost = miss_mult x the
        # normal max cost. Keeps a pose that shoves the scan off the map / into
        # never-observed space from ever competing with a real fit.
        self.miss_mult = float(miss_mult)
        # Depth pixels are NOT independent (thousands come off the same wall), so
        # summing every point overstates the evidence and the estimate over-fits
        # the scan. Normalize to an effective beam count instead.
        self.n_eff = float(n_eff)
        # Standard likelihood field: distance (m) from each cell to the nearest
        # obstacle. A depth ray stops at the FIRST surface, so a scan point that
        # lands anywhere ON an obstacle is a hit -> distance 0 -> cost 0, no matter
        # how thick the obstacle. distanceTransform(free) already returns 0 for
        # every obstacle cell and the distance-to-nearest-obstacle for free cells.
        free_u8 = gmap.navigable.astype(np.uint8)
        self.dist = cv2.distanceTransform(free_u8, cv2.DIST_L2, 5) * gmap.res
        # Cells actually observed while building the map. A scan point landing
        # where nothing was ever observed carries NO evidence, so it must not be
        # rewarded for happening to sit near a mapped wall -> treated as a miss.
        self.seen = getattr(gmap, "seen", None)
        # Obstacle INTERIOR (boundary layer eroded). The free-space term counts only
        # these, so a ray merely GRAZING a wall/counter face (boundary cells) is not
        # penalised -- only rays that pass THROUGH an obstacle count. Without this a
        # true pose alongside a wall gets a large spurious free penalty.
        obs_u8 = (~gmap.navigable).astype(np.uint8)
        self.obst_interior = cv2.erode(obs_u8, np.ones((3, 3), np.uint8),
                                       iterations=1).astype(np.float64)
        # UNSEEN interior (boundary layer eroded): cone_unseen counts only these, so
        # a cone that merely grazes the edge of an unseen region is not penalised --
        # only cones reaching DEEP into never-mapped space count.
        if self.seen is not None:
            uns_u8 = (~self.seen).astype(np.uint8)
            self.unseen_interior = cv2.erode(uns_u8, np.ones((3, 3), np.uint8),
                                             iterations=1).astype(np.float64)
        else:
            self.unseen_interior = None
        # Height-aware near gate for the FREE term. The depth camera has a limited
        # vertical FOV, so within a near range it cannot see the bottom of the
        # obstacle band -> a low unseen obstacle is looked OVER and its cells get
        # wrongly declared free (spurious see-through-wall penalty at the true pose).
        # A free ray span is therefore only cast beyond a per-ray lower bound:
        #   "off"  -> no gating (0.15 m), the original behaviour;
        #   "geom" -> a single scalar gate_smin from the camera geometry;
        #   "data" -> per-bearing gate_slo[bearing_bin]: the nearest range at which
        #             the frame actually observed a point at/below the band bottom.
        self.free_gate_mode = "off"
        self.gate_smin = 0.15
        self.gate_slo = {}

    def set_free_gate(self, mode="off", smin=0.15, slo=None):
        """Configure the height-aware near gate for the free term (see __init__)."""
        self.free_gate_mode = mode
        self.gate_smin = float(smin)
        self.gate_slo = slo or {}

    def score_world(self, pts_world):
        """pts_world: (..., N, 2) [x, z] -> LOG-likelihood per pose (...,).

        Sum of per-point log terms (independent beams). A *mean of exp()* saturates
        near 1 once most points are within sigma and then cannot resolve a 5 cm
        offset; summing log terms accumulates those small differences over ~1000
        points and stays discriminative. The distance is capped at d_max so an
        unmapped object contributes a bounded penalty instead of dominating the
        fit the way a plain least-squares/ICP cost would.

        A point placed OUTSIDE the map or in an UNSEEN cell gets the maximum
        (d_max) penalty: previously such points were clamped to the nearest border
        cell and read its distance, so a pose that shoved the scan off the map edge
        (or into never-observed space) next to a wall was scored as a good fit.
        """
        g = self.g
        col = (pts_world[..., 0] - g.min_x) / g.res
        row = (pts_world[..., 1] - g.min_z) / g.res
        ri = np.round(row).astype(np.int32)
        ci = np.round(col).astype(np.int32)
        oob = (ri < 0) | (ri >= g.H) | (ci < 0) | (ci >= g.W)   # off the map
        r = np.clip(ri, 0, g.H - 1)
        c = np.clip(ci, 0, g.W - 1)
        dd = self.dist[r, c]
        d = np.minimum(dd, self.d_max)
        cost = (d ** 2) / (2.0 * self.sigma ** 2)
        miss = self.miss_mult * (self.d_max ** 2) / (2.0 * self.sigma ** 2)
        # Hard miss for OFF-MAP, or OPEN unseen (unseen AND far from any mapped
        # obstacle, dist >= d_max). An unseen cell within d_max of a wall is
        # explained by that (seen) wall and keeps the normal distance cost -- so a
        # wall endpoint whose far side is unseen isn't flipped to a huge penalty by
        # a rounding nudge.
        bad = oob
        if self.seen is not None:
            bad = bad | (~oob & ~self.seen[r, c] & (dd >= self.d_max))
        cost = np.where(bad, miss, cost)
        if self.trim < 1.0 and cost.shape[-1] > 4:
            k = max(4, int(cost.shape[-1] * self.trim))
            cost = np.sort(cost, axis=-1)[..., :k]   # keep the best (drop outliers)
        return -cost.mean(axis=-1) * self.n_eff

    def cost_map(self):
        """Per-cell cost = cost(dist), with UNSEEN cells set to the (large) miss
        cost. Returns (cost_map, miss_cost); miss_cost is also the pad value used
        by score_field so off-map offsets are the same hard miss.
        """
        miss = self.miss_mult * (self.d_max ** 2) / (2.0 * self.sigma ** 2)
        cm = (np.minimum(self.dist, self.d_max) ** 2) / (2.0 * self.sigma ** 2)
        if self.seen is not None:                       # only OPEN unseen -> miss
            cm = np.where(~self.seen & (self.dist >= self.d_max), miss, cm)
        return cm, miss

    def score_field(self, scan, yaw, free_lambda=0.0, unseen_lambda=0.0):
        """Log-likelihood of EVERY map cell for a FIXED yaw, in one shot.

        For a fixed heading the per-cell score is a cross-correlation of the cost
        map with the scan placed as an impulse kernel (one impulse per point):
            score(X,Z) = -(n_eff/N) * Σ_i costmap(X+dx_i, Z+dz_i)
        Off-map offsets read the max-penalty padding, so points pushed off the map
        (or into unseen cells) are penalised exactly as in score_world. Evaluated
        at cell centres this is numerically identical to looping score_world over
        every cell, but ~100x faster (all points, FFT). No per-pose trim.

        free_lambda > 0 adds the FREE-SPACE term (see free_field): each ray is free
        up to its endpoint, so any mapped obstacle lying on that free span means the
        pose "sees through a wall" -> penalty. This kills endpoint-only aliases that
        fit walls but would require looking through other walls.
        Returns an (H, W) array. (Rotation convention matches score_world/rot2d.)
        """
        from scipy.signal import fftconvolve
        g = self.g
        if scan is None or len(scan) < 1:
            return np.full((g.H, g.W), -np.inf)
        cm, miss = self.cost_map()
        ca, sa = np.cos(yaw), np.sin(yaw)
        rx = scan[:, 0] * ca - scan[:, 1] * sa
        rz = scan[:, 0] * sa + scan[:, 1] * ca
        dcx = np.round(rx / g.res).astype(int)      # integer cell offsets
        dcz = np.round(rz / g.res).astype(int)
        kr = int(max(1, np.abs(dcx).max(), np.abs(dcz).max()))
        K = np.zeros((2 * kr + 1, 2 * kr + 1))
        np.add.at(K, (kr + dcz, kr + dcx), 1.0)     # impulse per scan point
        N = float(K.sum())
        padded = np.pad(cm, kr, mode="constant", constant_values=miss)   # off-map = miss
        corr = fftconvolve(padded, K[::-1, ::-1], mode="valid")   # Σ costmap(pos+off)
        score = -(self.n_eff / N) * corr
        if free_lambda > 0.0:
            Kf, kr, nf = self._free_kernel(scan, yaw)
            if nf > 0:
                score = score - free_lambda * (self.n_eff / nf) * self._corr_free(
                    self.obst_interior, Kf, kr, pad=0.0)
        if unseen_lambda > 0.0 and self.seen is not None:
            Kf, kr, nf = self._free_kernel(scan, yaw, nearest=False)   # farthest extent
            ksum = float(Kf.sum())
            if ksum > 0:                     # fraction of cone reaching DEEP unseen
                score = score - unseen_lambda * self.n_eff * self._corr_free(
                    self.unseen_interior, Kf, kr, pad=0.0) / ksum
        return score

    def _free_kernel(self, scan, yaw, nearest=True):
        """Rasterised span impulse kernel for the FOV cone. nearest=True casts each
        bearing up to its FIRST return (free/see-through-wall check); nearest=False
        casts up to the FARTHEST return (full observed extent, for cone-unseen).
        Returns (kernel, radius, n_rays)."""
        g = self.g
        sx, sz = scan[:, 0], scan[:, 1]
        rngm = np.hypot(sx, sz) + 1e-9
        b = np.round(np.arctan2(sx, -sz) / np.deg2rad(0.5)).astype(int)
        ub, inv = np.unique(b, return_inverse=True)     # one ray per bearing
        if nearest:
            agg = np.full(len(ub), np.inf); np.minimum.at(agg, inv, rngm)
            keep = rngm <= agg[inv] + 1e-6              # nearest (first hit)
        else:
            agg = np.zeros(len(ub)); np.maximum.at(agg, inv, rngm)
            keep = rngm >= agg[inv] - 1e-6              # farthest return
        sx, sz, rngm = sx[keep], sz[keep], rngm[keep]
        ca, sa = np.cos(yaw), np.sin(yaw)
        rx = sx * ca - sz * sa
        rz = sx * sa + sz * ca
        ux, uz = rx / rngm, rz / rngm
        # per-ray near lower bound: the free span starts only where the camera could
        # actually see down to the obstacle band bottom (height-aware gate). Applies
        # to the FREE term only (nearest=True); the cone-unseen extent is unaffected.
        if nearest and self.free_gate_mode == "geom":
            lo = np.full(sx.shape, self.gate_smin)
        elif nearest and self.free_gate_mode == "data":
            bk = np.round(np.arctan2(sx, -sz) / np.deg2rad(0.5)).astype(int)
            lo = np.array([self.gate_slo.get(int(b), np.inf) for b in bk])
        else:
            lo = np.full(sx.shape, 0.15)
        lo = np.maximum(lo, 0.15)
        S = np.arange(0.15, float(rngm.max()), g.res)
        # free span: from the per-ray near gate up to just before the first hit
        valid = (S[None, :] >= lo[:, None]) & (S[None, :] < (rngm[:, None] - 0.10))
        px = (ux[:, None] * S[None, :])[valid]
        pz = (uz[:, None] * S[None, :])[valid]
        if px.size == 0:                     # no free span (tiny / all-clamped scan)
            return np.zeros((3, 3)), 1, 0
        dcx = np.round(px / g.res).astype(int)
        dcz = np.round(pz / g.res).astype(int)
        kr = int(max(1, np.abs(dcx).max(), np.abs(dcz).max()))
        Kf = np.zeros((2 * kr + 1, 2 * kr + 1))
        np.add.at(Kf, (kr + dcz, kr + dcx), 1.0)
        return Kf, kr, int(keep.sum())

    @staticmethod
    def _corr_free(indicator, Kf, kr, pad=0.0):
        from scipy.signal import fftconvolve
        padded = np.pad(indicator, kr, mode="constant", constant_values=pad)
        return fftconvolve(padded, Kf[::-1, ::-1], mode="valid")

    @staticmethod
    def _grouped_corr(m, kernels, krs, pad):
        """Cross-correlate a FIXED map `m` with per-yaw `kernels`, computing the map FFT
        ONCE per DISTINCT kr and reusing it for every yaw that shares that kr. Each yaw's
        kernel FFT + inverse is done separately (multithreaded), reproducing
        scipy.fftconvolve(pad(m,kr,pad), K[::-1,::-1], 'valid') EXACTLY -- same fshape
        (next_fast_len), same dtype/precision (m's dtype: cost map float32 -> complex64,
        interior float64 -> complex128), same 'valid' crop. So the result is BIT-IDENTICAL
        (np.array_equal) to the per-yaw loop; the only saved work is re-FFT'ing the fixed
        map. (Per-yaw beats a batched transform here: batching allocates a Y x fshape
        complex volume; per-yaw stays cache-resident and multithreads each 2-D FFT.
        float32 transforms ARE ~1.4x faster but compound to -4 episodes of SR over full
        runs, with no total-eval gain -- the FFT is not the eval bottleneck -- so keep f64.)"""
        import os
        import scipy.fft as sfft
        w = max(1, os.cpu_count() or 1)
        Y = len(kernels)
        H, W = m.shape
        out = np.empty((Y, H, W), np.float64)
        krs = np.asarray(krs, dtype=int)
        for kr in np.unique(krs):
            idx = np.nonzero(krs == kr)[0]
            kb = 2 * int(kr) + 1
            padded = np.pad(m, int(kr), mode="constant", constant_values=pad)  # keep m dtype
            fshape = [sfft.next_fast_len(padded.shape[0] + kb - 1, real=True),
                      sfft.next_fast_len(padded.shape[1] + kb - 1, real=True)]
            Mf = sfft.rfftn(padded, fshape, workers=w)              # map FFT: once per kr
            r0 = kb - 1                                             # 'valid' crop origin
            for j in idx:
                Kf = sfft.rfftn(kernels[j][::-1, ::-1], fshape, workers=w)
                full = sfft.irfftn(Mf * Kf, fshape, workers=w)
                out[j] = full[r0:r0 + H, r0:r0 + W]
        return out

    def score_fields_batch(self, scan, yaws, free_lambda=0.0, unseen_lambda=0.0):
        """score_field for MANY yaws at once -> (len(yaws), H, W). BIT-IDENTICAL to
        looping score_field per yaw (endpoint + free + cone-unseen): the same per-yaw
        kernels/kr are built, then _grouped_corr shares the map FFT across yaws that
        share a kr (see _grouped_corr) instead of one fftconvolve per yaw."""
        g = self.g
        yaws = np.asarray(yaws, dtype=float)
        Y = len(yaws)
        if scan is None or len(scan) < 1:
            return np.full((Y, g.H, g.W), -np.inf)
        cm, miss = self.cost_map()

        # --- endpoint (obstacle) term: per-yaw impulse kernel with its own tight kr ---
        ca = np.cos(yaws)[:, None]; sa = np.sin(yaws)[:, None]
        dcx = np.round((scan[:, 0] * ca - scan[:, 1] * sa) / g.res).astype(int)   # (Y,N)
        dcz = np.round((scan[:, 0] * sa + scan[:, 1] * ca) / g.res).astype(int)
        K_list, kr_list = [], []
        for y in range(Y):
            kr = int(max(1, np.abs(dcx[y]).max(), np.abs(dcz[y]).max()))
            K = np.zeros((2 * kr + 1, 2 * kr + 1))
            np.add.at(K, (kr + dcz[y], kr + dcx[y]), 1.0)
            K_list.append(K); kr_list.append(kr)
        N = float(len(scan))
        score = -(self.n_eff / N) * self._grouped_corr(cm, K_list, kr_list, miss)

        # --- free-space (see-through-wall) term (per-yaw _free_kernel, unchanged) ---
        # scaling matches the per-yaw score_field's exact float association --
        # free_lambda*(n_eff/nf) -- so the sum is bit-identical, not just ~1e-13 close.
        if free_lambda > 0.0:
            kfs = [self._free_kernel(scan, ya, nearest=True) for ya in yaws]
            corr = self._grouped_corr(self.obst_interior, [k[0] for k in kfs],
                                      [k[1] for k in kfs], 0.0)
            nf = np.array([k[2] for k in kfs], dtype=float)
            nf_safe = np.where(nf > 0, nf, 1.0)
            coef = np.where(nf > 0, free_lambda * (self.n_eff / nf_safe), 0.0)
            score = score - coef[:, None, None] * corr

        # --- cone-unseen term (match association: (unseen_lambda*n_eff)*corr/ksum) ---
        if unseen_lambda > 0.0 and self.seen is not None:
            kfs = [self._free_kernel(scan, ya, nearest=False) for ya in yaws]
            corr = self._grouped_corr(self.unseen_interior, [k[0] for k in kfs],
                                      [k[1] for k in kfs], 0.0)
            ksum = np.array([float(k[0].sum()) for k in kfs], dtype=float)
            ksum_safe = np.where(ksum > 0, ksum, 1.0)
            term = (unseen_lambda * self.n_eff) * corr / ksum_safe[:, None, None]
            score = score - np.where(ksum[:, None, None] > 0, term, 0.0)

        return score

    def free_field(self, scan, yaw):
        """Per-cell count of obstacle INTERIOR cells on the FOV free span
        (see-through-wall penalty; grazing a wall face does not count).
        off-map = not an obstacle -> 0."""
        Kf, kr, _ = self._free_kernel(scan, yaw)
        return self._corr_free(self.obst_interior, Kf, kr, pad=0.0)

    def cone_unseen_field(self, scan, yaw):
        """Per-cell count of DEEP-UNSEEN cells inside the FOV cone. The cone extends
        to the FARTHEST return per bearing (full observed extent) and only unseen
        INTERIOR cells count (grazing the edge of unseen does not). A pose whose cone
        reaches into never-mapped space is implausible (the map came from real
        traversals). off-map -> 0 (handled by the endpoint miss term)."""
        if self.seen is None:
            return np.zeros((self.g.H, self.g.W))
        Kf, kr, _ = self._free_kernel(scan, yaw, nearest=False)     # farthest extent
        return self._corr_free(self.unseen_interior, Kf, kr, pad=0.0)

    def score_breakdown(self, scan, r, c, yaw, free_lambda, unseen_lambda):
        """Decompose the 3-term score at one cell/yaw into (endpoint E, free penalty,
        cone-unseen penalty, total). Uses the SAME fields and guards as score_field,
        so this is the single source for both scoring and the plots' per-pose numbers.
        total = E - free - cone (score_field subtracts the two penalties)."""
        E = float(self.score_field(scan, yaw)[r, c])
        fpen = 0.0
        if free_lambda > 0.0:
            _, _, nf = self._free_kernel(scan, yaw)
            if nf > 0:
                fpen = free_lambda * (self.n_eff / nf) * float(
                    self.free_field(scan, yaw)[r, c])
        upen = 0.0
        if unseen_lambda > 0.0 and self.seen is not None:
            Kfc, _, _ = self._free_kernel(scan, yaw, nearest=False)
            ncone = float(Kfc.sum())
            if ncone > 0:
                upen = unseen_lambda * self.n_eff * float(
                    self.cone_unseen_field(scan, yaw)[r, c]) / ncone
        return {"E": E, "free": fpen, "cone": upen, "total": E - fpen - upen}


class ParticleFilterLocalizer:
    """Main localizer: global particle filter (Monte-Carlo localization) with
    SINGLE-FRAME scan measurements and cumulative weights.

    Where the legacy ``AccumGridParticleFilter`` accumulates several scans into a
    locates the pose with a single dense grid sweep, this follows the classic
    recursive Monte-Carlo filter and the reference ``SimpleParticles`` pipeline --
    no scan buffer, no sweep:

      init    : dense particles over every navigable cell x a fine yaw grid
      predict : move each particle by the executed action + per-particle noise
      weight  : score each particle against the CURRENT single frame's scan only
      accum   : before convergence, add capped per-frame log-likelihoods so a
                particle that explains every frame dominates. After convergence,
                retain the local posterior mode and average each normalized frame
                weight with a bounded prior-frame inertia
      resample: systematic, but only once enough motion (hence new evidence) has
                accrued; post-convergence resampling restores the tracking inertia
      estimate: weighted mean of the whole particle cloud

    The estimate is meaningless until the cloud collapses (the mean of a map-wide
    cloud sits in a wall), so like the other global localizers this exposes
    ``converged``. The pre-convergence policy gathers scan and motion evidence until
    the configured spread, translation, yaw, and information gates are satisfied.
    """

    def __init__(self, *, fx, fy, cx, cy, max_depth, sensor_height,
                 sensor_position=None,
                 img_height=None, tilt_deg=0.0, h_min=0.2, h_max=1.5,
                 max_range=5.0, subsample=8, forward_step=0.25, turn_angle=30.0,
                 actions=None,
                 free_gate="geom", free_gate_scale=0.8,
                 angle_bin=72, pos_sparsity=1,
                 mcl_trans_noise=0.02, mcl_rot_noise_deg=3.0,
                 resample_trans=0.35, resample_rot_deg=180.0,
                 rough_xy=0.05, rough_yaw_deg=4.0,
                 sigma=0.25, n_eff_mcl=25.0, max_scan_points=None,
                 global_loglik_cap=10.0,
                 converge_std=0.5, converge_hold=5, min_converge_trans=0.0,
                 min_converge_yaw_deg=180.0,
                 inject_frac=0.0, max_particles=3_000_000,
                 free_lambda=3.0, unseen_lambda=1.0,
                 info_gating=True, info_tau=0.35, info_scale=0.15,
                 info_temper=True, info_beta_floor=0.15,
                 require_info_resample=True, min_info_frames=3,
                 ambig_inject_frac=0.05,
                 divergence_recovery=True, div_gap=80.0, div_hold=6,
                 div_check_every=1, div_max_recover=3, active_seek=True,
                 div_local=False, div_local_radius=1.5,
                 cumulative_conf=True, cum_init_cnt=10,
                 tracking_radius_m=0.75,
                 settle_std=0.20, settle_hold=8,
                 accum_scan=False, accum_scan_window=8,
                 seed=0):
        # --- camera intrinsics + motion model (merged from the old BaseLocalizer) --
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        # image height (px) for the exact vertical-FOV derivation; falls back to
        # 2*cy (centered principal point) when the caller doesn't supply it.
        self.img_height = float(img_height) if img_height is not None else 2.0 * float(cy)
        self.max_depth = float(max_depth)
        self.sensor_height = float(sensor_height)
        self.sensor_position = np.asarray(
            [0.0, self.sensor_height, 0.0]
            if sensor_position is None else sensor_position,
            dtype=np.float64,
        )
        if self.sensor_position.shape != (3,):
            raise ValueError("sensor_position must be [right, height, backward] in metres")
        self.tilt = np.deg2rad(float(tilt_deg))
        self.h_min, self.h_max = float(h_min), float(h_max)
        self.max_range = float(max_range)
        self.subsample = int(subsample)
        self.forward_step = float(forward_step)
        self.turn = np.deg2rad(float(turn_angle))
        # action ids: {"stop","move_forward","turn_left","turn_right"}
        self.A = actions or {"stop": 0, "move_forward": 1, "turn_left": 2, "turn_right": 3}
        # height-aware near gate for the free term (see LikelihoodField.set_free_gate):
        # "geom" cuts the free span within s_min of the camera; scale<1 relaxes it
        self.free_gate = str(free_gate)
        self.free_gate_scale = float(free_gate_scale)
        self.lf = None
        self.x = self.z = self.yaw = 0.0
        self.y = 0.0
        # --- particle-filter parameters ---
        self.angle_bin = int(angle_bin)              # 36 -> 10 deg, 72 -> 5 deg
        self.pos_sparsity = int(pos_sparsity)        # keep 1/N of the free cells
        self.mcl_trans_noise = float(mcl_trans_noise)
        self.mcl_rot_noise = np.deg2rad(float(mcl_rot_noise_deg))
        self.resample_trans = float(resample_trans)
        self.resample_rot = np.deg2rad(float(resample_rot_deg))
        self.rough_xy = float(rough_xy)
        self.rough_yaw = np.deg2rad(float(rough_yaw_deg))
        self.sigma = float(sigma)
        self.n_eff_mcl = float(n_eff_mcl)            # per-frame likelihood sharpness
        # During global search, no finite particle may lose more than this many
        # log-likelihood units to the frame winner in ONE observation. Narrow-FOV
        # frames overlap heavily during a spin; without this bound, multiplying
        # their correlated likelihoods can irreversibly erase the true map mode.
        # None (or <= 0) restores the historical unbounded product.
        self.global_loglik_cap = (None if global_loglik_cap is None
                                  else float(global_loglik_cap))
        # None -> use EVERY scan point. score_field is an FFT convolution whose cost
        # is independent of the point count (all points just fill one kernel), so
        # capping the scan throws away evidence for no speed gain.
        self.max_scan_points = int(max_scan_points) if max_scan_points else None
        self.converge_std = float(converge_std)
        self.converge_hold = int(converge_hold)
        # A map-wide cloud can look "converged" before it has observed enough distinct
        # viewpoints. Require both translation and accumulated absolute yaw motion;
        # callers that localize in place can set min_converge_trans=0 while retaining
        # the yaw gate.
        self.min_converge_trans = float(min_converge_trans)
        self.min_converge_yaw = np.deg2rad(float(min_converge_yaw_deg))
        # Augmented-MCL recovery: sprinkle a few uniform particles over the map at
        # every resample. Without it, once the true region loses all its particles
        # (early wrong collapse) roughening cannot bring them back and the filter is
        # stuck for good -- the main closed-loop failure mode of single-frame MCL.
        self.inject_frac = float(inject_frac)
        self.max_particles = int(max_particles)
        # ray-consistency penalties -- SAME likelihood as the plots / score_field
        self.free_lambda = float(free_lambda)
        self.unseen_lambda = float(unseen_lambda)
        # --- ambiguity handling (opt-in EXTENSION; info_gating=False -> original) ---
        # A single frame facing a blank wall matches many poses, so the cumulative
        # filter can collapse onto a WRONG mode before an informative view arrives.
        # These gates temper uninformative frames' weight, delay resampling and
        # convergence until enough informative frames are seen, and inject more
        # uniform hypotheses while still ambiguous (so an early wrong collapse can
        # recover). All are no-ops when info_gating=False (exact prior behaviour).
        self.info_gating = bool(info_gating)
        self.info_tau = float(info_tau)          # frame "informative" if info >= tau
        self.info_scale = float(info_scale)      # maps scan 2-D spread ratio -> [0,1]
        self.info_temper = bool(info_temper)
        self.info_beta_floor = float(info_beta_floor)
        self.require_info_resample = bool(require_info_resample)
        self.min_info_frames = int(min_info_frames)
        self.ambig_inject_frac = float(ambig_inject_frac)
        self._base_inject = self.inject_frac     # restore target once informative
        self._n_info = 0
        self._last_info = 0.0
        # kidnapped-robot recovery (EXTENSION): once converged, a confident-WRONG
        # lock (global aliasing) shows up as the single-frame MAP estimate sitting
        # persistently FAR from the filter estimate. Detect that and re-globalize.
        self.divergence_recovery = bool(divergence_recovery)
        # gap = (global-best single-frame score) - (score AT the estimate). A correct
        # lock explains the frame (gap small); a wrong/aliased lock does not (gap huge:
        # measured ~0-12 on good locks vs ~60-375 on wrong locks). Default raised to
        # gap=80/hold=6 (from 30/3) so the recovery does NOT false-fire on a good
        # lock and re-globalize it -- that showed up as a post-convergence 'jump'.
        self.div_gap = float(div_gap)
        self.div_hold = int(div_hold)            # consecutive informative checks
        self.div_check_every = int(div_check_every)
        self.div_max_recover = int(div_max_recover)
        # local re-search: on divergence, re-scatter particles only within this radius
        # (m) of the current estimate instead of the whole map -> bounds the jump.
        self.div_local = bool(div_local)
        self.div_local_radius = float(div_local_radius)
        self.active_seek = bool(active_seek)     # Direction-1 (suggest_action) toggle
        # cumulative confidence (reference SimpleParticles style): once CONVERGED,
        # accumulate each frame's NORMALIZED weight additively (running average with
        # growing inertia cnt) instead of multiplying log-likelihoods. A single bad
        # frame then only nudges an established lock (bounded add) instead of
        # collapsing a particle's weight (product) and yanking the estimate. The
        # multiplicative path is kept for fast global convergence before the lock.
        self.cumulative_conf = bool(cumulative_conf)
        self.cum_init_cnt = int(cum_init_cnt)    # prior-frame count at lock / after resampling
        if self.cum_init_cnt < 0:
            raise ValueError("cum_init_cnt must be non-negative")
        # Once the global posterior has satisfied the convergence gates, discard
        # map-wide residual hypotheses before entering tracking. Keeping those
        # particles allows a later aliased single frame to resurrect another room.
        self.tracking_radius_m = float(tracking_radius_m)
        if self.tracking_radius_m <= 0.0:
            raise ValueError("tracking_radius_m must be positive")
        self._cum_cnt = 0
        self._cum_on = False
        # SETTLED: a stricter, SUSTAINED convergence (spread < settle_std for
        # settle_hold consecutive steps) used to gate trusting the pose downstream
        # (e.g. the dynamic map). converged flips on a loose/early lock; settled
        # waits until the cloud has actually stayed tight, so early updates on a
        # still-drifting pose don't smear the map.
        self.settle_std = float(settle_std)
        self.settle_hold = int(settle_hold)
        self.settled = False
        self._settle_cnt = 0
        # OPT-IN accumulated-scan likelihood (default off): score a windowed local submap --
        # the last `accum_scan_window` frames' obstacle points fused in an internal odometry
        # frame -- against the map instead of a single scan. More geometry disambiguates
        # feature-poor frames; cost is bounded by max_scan_points. See _accumulate().
        self.accum_scan = bool(accum_scan)
        self.accum_scan_window = max(1, int(accum_scan_window))
        self._acc_pts = deque(maxlen=self.accum_scan_window)   # obstacle pts in the odo frame
        self._acc_ox = self._acc_oz = self._acc_oyaw = 0.0     # internal dead-reckon odometry
        self._div_bad = 0
        self._steps = 0
        self._n_recover = 0
        self._total_yaw = 0.0
        self.rng = np.random.RandomState(seed)
        self.chunk = 8192
        self.converged = False
        self.spread = 1e9

    # ------------------------------------------------------------------ #
    # ---------------- camera / motion model (merged from BaseLocalizer) --------
    @property
    def position(self):
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def heading(self):
        return self.yaw

    def pose(self):
        return self.position, self.yaw

    def to_world(self, pts_robot, x, z, yaw):
        """(N,2) robot-frame [x,z] -> world, for pose (x, z, yaw)."""
        return rot2d(pts_robot, yaw) + np.array([x, z])

    def extract_scan(self, obs, depth_key="depth"):
        """depth image -> (N,2) obstacle points [x, z] in the robot frame."""
        if depth_key not in obs:
            return np.zeros((0, 2))
        dm = np.asarray(obs[depth_key]).squeeze() * self.max_depth
        s = self.subsample
        dm = dm[::s, ::s]
        H, W = dm.shape
        vs, us = np.mgrid[0:H, 0:W]
        us = us.reshape(-1) * s
        vs = vs.reshape(-1) * s
        z = dm.reshape(-1)
        ok = (z > 0.15) & (z < min(self.max_range, self.max_depth - 0.1))
        if ok.sum() < 10:
            return np.zeros((0, 2))
        # Camera pose expressed in the ROBOT frame. Keeping the full mounting
        # translation makes the particle pose represent the base rotation centre,
        # not an optical centre that moves around it during turns.
        pts = backproject_pixels(us[ok], vs[ok], z[ok], self.fx, self.fy, self.cx, self.cy,
                                 self.sensor_position,
                                 np.array([-self.tilt, 0.0, 0.0]))
        h = pts[:, 1]
        band = (h > self.h_min) & (h < self.h_max)
        if band.sum() < 10:
            return np.zeros((0, 2))
        return np.stack([pts[band, 0], pts[band, 2]], axis=-1)

    def _motion_delta(self, action, collided):
        """(d_forward, d_yaw) actually executed, per the motion model."""
        if action == self.A["move_forward"]:
            return (0.0 if collided else self.forward_step), 0.0
        # Verified against the sim: with yaw = atan2(fwd_x, -fwd_z), turn_left
        # DECREASES yaw and turn_right increases it.
        if action == self.A["turn_left"]:
            return 0.0, -self.turn
        if action == self.A["turn_right"]:
            return 0.0, self.turn
        return 0.0, 0.0

    def predict(self, action=None, collided=False, delta_pose=None):
        """Motion update (dead-reckon the point estimate, then move the cloud).

        Two input forms, mirroring the reference filter's ``run_particle_step``:
          * continuous ``delta_pose=(d_fwd, d_lat, dyaw)`` -- a relative pose change
            in the ROBOT frame (forward, left/right, yaw), as in the reference; or
          * a discrete ``action`` id (the SPECIAL CASE) -- mapped by the dead-reckoning
            motion model to ``delta_pose = (d, 0, dyaw)`` (forward_step / turn_angle),
            i.e. exactly the previous behaviour.

        Unlike before there is no early-out for a zero delta: in noise mode
        (``mcl_*_noise > 0``) the per-particle noise in ``_on_predict`` is applied on
        EVERY call, and it spreads BOTH robot-frame translation axes (forward AND
        lateral -> 2-D world x/z), not only the heading axis.
        """
        if delta_pose is None:
            d, dyaw = self._motion_delta(action, collided)
            d_fwd, d_lat = d, 0.0
        else:
            d_fwd, d_lat, dyaw = (float(v) for v in delta_pose)
        # dead-reckoned point estimate (no per-particle noise here). Robot-frame
        # forward = (sin yaw, -cos yaw); lateral = (cos yaw, sin yaw).
        self.yaw = wrap_pi(self.yaw + dyaw)
        self.x += d_fwd * np.sin(self.yaw) + d_lat * np.cos(self.yaw)
        self.z += -d_fwd * np.cos(self.yaw) + d_lat * np.sin(self.yaw)
        self._on_predict(d_fwd, d_lat, dyaw)
        if self.accum_scan:                     # advance the internal odometry for the submap
            self._acc_oyaw = wrap_pi(self._acc_oyaw + dyaw)
            self._acc_ox += d_fwd * np.sin(self._acc_oyaw) + d_lat * np.cos(self._acc_oyaw)
            self._acc_oz += -d_fwd * np.cos(self._acc_oyaw) + d_lat * np.sin(self._acc_oyaw)

    def reset(self, gmap):
        self._acc_pts.clear(); self._acc_ox = self._acc_oz = self._acc_oyaw = 0.0
        # match the depth-scan height band to the one the map was built with, so
        # scan points and mapped geometry come from the SAME slice of the world
        hmin = getattr(gmap, "obstacle_height_min", None)
        hmax = getattr(gmap, "obstacle_height_max", None)
        if hmin is not None and hmax is not None:
            self.h_min, self.h_max = float(hmin), float(hmax)
        # single-frame likelihood field: soft sigma / n_eff so one ambiguous frame
        # can't collapse the cloud onto a lucky particle; trim off (use all points)
        self.lf = LikelihoodField(gmap, sigma_m=self.sigma, trim=1.0, n_eff=self.n_eff_mcl)
        if self.free_gate == "geom":
            # camera vertical half-angle downward (from intrinsics -> follows the
            # sensor config); below s_min the FOV can't reach the obstacle-band
            # bottom, so free there is unreliable. scale relaxes the hard cut.
            theta_down = vertical_half_fov_down(self.fy, self.cy, self.img_height)
            s_min = (self.sensor_height - self.h_min) / np.tan(theta_down)
            self.lf.set_free_gate("geom", smin=self.free_gate_scale * s_min)
        # unknown start: the pose estimate is undefined until the first scan update
        # overwrites it (particles are seeded map-wide below, not around a start pose).
        self.x = self.y = self.z = 0.0
        self.yaw = 0.0
        self._gmap = gmap

        # start pose is UNKNOWN (the only supported mode): dense particles over every
        # EXPLORED free cell x fine yaw (seen mask -- don't seed hypotheses in
        # never-observed space), then spin/search until the cloud collapses.
        fr, fc = np.nonzero(getattr(gmap, "explorable", gmap.free))
        sel = np.arange(0, len(fr), self.pos_sparsity)
        gx = fc[sel] * gmap.res + gmap.min_x
        gz = fr[sel] * gmap.res + gmap.min_z
        yaws = np.arange(self.angle_bin) * (2 * np.pi / self.angle_bin) - np.pi
        n_pos = len(gx)
        if n_pos * self.angle_bin > self.max_particles:   # cap for speed
            keep = max(1, self.max_particles // self.angle_bin)
            idx = self.rng.choice(n_pos, keep, replace=False)
            gx, gz, n_pos = gx[idx], gz[idx], keep
        P = np.zeros((n_pos * self.angle_bin, 3))
        P[:, 0] = np.repeat(gx, self.angle_bin)
        P[:, 1] = np.repeat(gz, self.angle_bin)
        P[:, 2] = np.tile(yaws, n_pos)
        self.converged = False
        self.particles = P
        self.P = len(P)
        self.logw = np.zeros(self.P)                 # cumulative log-likelihood
        self.weights = np.full(self.P, 1.0 / self.P)
        self._accum_trans = 0.0
        self._accum_rot = 0.0
        self._total_trans = 0.0
        self._total_yaw = 0.0
        self.spread = 1e9
        self._hold = 0
        self._n_info = 0             # informative frames seen this episode
        self._last_info = 0.0
        self.inject_frac = self._base_inject
        self._cum_cnt = 0
        self._cum_on = False
        self.settled = False
        self._settle_cnt = 0
        self._div_bad = 0
        self._steps = 0
        self._n_recover = 0

    # ------------------------------------------------------------------ #
    def _frame_info(self, scan):
        """Informativeness of a SINGLE frame in [0,1]: how 2-D (localizing) its scan
        is. A blank wall -> points lie on a line (aliases along every wall) -> ~0; a
        corner / room with >=2 surface directions -> spread in two dims -> higher.
        Cheap PCA proxy (no extra field); only gates the ambiguity extension."""
        if scan is None or len(scan) < 20:
            return 0.0
        c = scan - scan.mean(axis=0)
        cov = (c.T @ c) / len(c)
        ev = np.linalg.eigvalsh(cov)             # ascending [lam_min, lam_max]
        lam_min, lam_max = float(ev[0]), float(ev[-1])
        shape = lam_min / (lam_max + 1e-9)       # 0 = line (wall), larger = 2-D
        return float(np.clip(shape / self.info_scale, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    def _on_predict(self, d_fwd, d_lat, dyaw):
        """Move the whole cloud by the (robot-frame) delta + per-particle noise.

        Per-particle Gaussian noise is added to the yaw and to BOTH robot-frame
        translation axes (forward + lateral); once rotated into the world this
        spreads x AND z (2-D), not only the heading axis. When a noise std is > 0
        the noise is applied unconditionally -- also for a zero delta -- so the
        cloud always diffuses in noise mode (matches reference add_noise, which
        noises dx and dy of every delta). accum_* track only the REAL motion."""
        p = self.particles
        n = self.P
        dyaw_p = (dyaw + self.rng.randn(n) * self.mcl_rot_noise
                  if self.mcl_rot_noise > 0 else dyaw)
        p[:, 2] = wrap_pi(p[:, 2] + dyaw_p)
        if self.mcl_trans_noise > 0:
            fwd = d_fwd + self.rng.randn(n) * self.mcl_trans_noise
            lat = d_lat + self.rng.randn(n) * self.mcl_trans_noise
        else:
            fwd, lat = d_fwd, d_lat
        ca, sa = np.cos(p[:, 2]), np.sin(p[:, 2])
        p[:, 0] += fwd * sa + lat * ca            # forward=(sin,-cos), lateral=(cos,sin)
        p[:, 1] += -fwd * ca + lat * sa
        self._accum_trans += abs(d_fwd) + abs(d_lat)
        self._accum_rot += abs(dyaw)
        self._total_trans += float(np.hypot(d_fwd, d_lat))
        self._total_yaw += abs(float(dyaw))

    def _accumulate(self, cur_scan):
        """OPT-IN (accum_scan): buffer THIS frame's obstacle points in the internal odometry
        frame, then return the last `accum_scan_window` frames' points transformed back into
        the CURRENT robot frame -- a small local submap to score in place of the single scan.
        Same odometry-frame convention as ShieldLocalMap. NOTE: consecutive frames overlap, so
        this re-uses correlated points across steps and the belief can become over-confident;
        it is opt-in and meant to be validated (SR/ATE) before being trusted."""
        ca, sa = np.cos(self._acc_oyaw), np.sin(self._acc_oyaw)
        if cur_scan is not None and len(cur_scan):
            wx = cur_scan[:, 0] * ca - cur_scan[:, 1] * sa + self._acc_ox
            wz = cur_scan[:, 0] * sa + cur_scan[:, 1] * ca + self._acc_oz
            self._acc_pts.append(np.stack([wx, wz], axis=1))
        if not self._acc_pts:
            return cur_scan
        P = np.concatenate(self._acc_pts, axis=0)           # odo-world points over the window
        dx = P[:, 0] - self._acc_ox; dz = P[:, 1] - self._acc_oz
        rx = dx * ca + dz * sa                              # -> current robot frame (x = right)
        rz = -dx * sa + dz * ca                             #    forward = -z
        return np.stack([rx, rz], axis=1)

    def _score_all(self, scan, vol=None):
        """Per-particle single-frame log-likelihood, IDENTICAL to the plots: the
        exact same LikelihoodField.score_field(scan, yaw, free_lambda, unseen_lambda)
        (endpoint + free-space + cone-unseen). Particles have continuous yaw, so bin
        them to `angle_bin` headings, compute one field per occupied bin (free/cone
        are global convolutions, so one FFT per yaw), and read each particle's cell.

        If `vol` (the full nb x H x W score volume for THIS scan -- e.g. the one the
        divergence-recovery check already builds via score_volume) is supplied, particles
        read their cell STRAIGHT from it, skipping the FFT here. Each per-yaw field is
        transformed independently, so vol[b] equals the occupied-subset field for bin b
        bit-for-bit -> the result is identical, only the duplicate transform is saved.
        """
        g = self._gmap
        nb = self.angle_bin
        step = 2 * np.pi / nb
        ybin = (np.round((self.particles[:, 2] + np.pi) / step).astype(int)) % nb
        col = np.clip(np.round((self.particles[:, 0] - g.min_x) / g.res).astype(int), 0, g.W - 1)
        row = np.clip(np.round((self.particles[:, 1] - g.min_z) / g.res).astype(int), 0, g.H - 1)
        if vol is not None:
            return vol[ybin, row, col]        # reuse the precomputed full volume: no FFT
        ub = np.unique(ybin)
        # one batched pass over the occupied yaw bins (fixed maps FFT'd once, kernels
        # batched) instead of an fftconvolve per bin -- same score, ~3-5x faster.
        volb = self.lf.score_fields_batch(scan, ub * step - np.pi,
                                          free_lambda=self.free_lambda,
                                          unseen_lambda=self.unseen_lambda)
        # vectorized readout: ub is sorted and every ybin is in it, so searchsorted gives
        # each particle's row in volb -> one gather instead of a per-bin Python loop.
        idx = np.searchsorted(ub, ybin)
        return volb[idx, row, col]

    def score_volume(self, scan):
        """Dense single-frame log-likelihood over EVERY map cell x self.angle_bin
        headings -- the SAME score (free/cone weights + gate) the particle filter
        uses. Returns (yaws (nb,), volume (nb, H, W)). This is the one function plots
        should call: they render it, they do not re-derive the yaw grid / weights."""
        nb = self.angle_bin
        yaws = np.arange(nb) * (2 * np.pi / nb) - np.pi
        vol = self.lf.score_fields_batch(scan, yaws, free_lambda=self.free_lambda,
                                         unseen_lambda=self.unseen_lambda)
        return yaws, vol

    def topk_poses(self, scan, gmap, k=5):
        """Top-k candidate poses over the whole (cell x yaw) volume, restricted to
        the explorable mask. Returns [(row, col, yaw, total_score), ...] best first."""
        yaws, vol = self.score_volume(scan)
        mask = getattr(gmap, "explorable", gmap.free)
        vol = vol.copy()
        vol[:, ~mask] = -np.inf
        flat = np.argsort(vol.ravel())[-k:][::-1]
        out = []
        for fi in flat:
            yi, r, c = np.unravel_index(fi, vol.shape)
            out.append((int(r), int(c), float(yaws[yi]), float(vol[yi, r, c])))
        return out

    def _estimate(self):
        w = self.weights
        x = float((self.particles[:, 0] * w).sum())
        z = float((self.particles[:, 1] * w).sum())
        yaw = float(np.arctan2((np.sin(self.particles[:, 2]) * w).sum(),
                               (np.cos(self.particles[:, 2]) * w).sum()))
        return x, z, yaw

    def _check_convergence(self):
        w = self.weights
        mx = (self.particles[:, 0] * w).sum()
        mz = (self.particles[:, 1] * w).sum()
        var = (w * ((self.particles[:, 0] - mx) ** 2
                    + (self.particles[:, 1] - mz) ** 2)).sum()
        self.spread = float(np.sqrt(max(var, 0.0)))
        ok = (self.spread < self.converge_std
              and self._total_trans >= self.min_converge_trans
              and self._total_yaw + 1e-9 >= self.min_converge_yaw
              # don't commit while only blank-wall frames have been seen (EXTENSION)
              and (not self.info_gating or self._n_info >= self.min_info_frames))
        self._hold = self._hold + 1 if ok else 0
        entered_tracking = False
        if not self.converged and self._hold >= self.converge_hold:
            self.converged = True
            self._enter_tracking(mx, mz)
            entered_tracking = True
        # sustained-tight gate (see __init__): only True after the lock has held
        if self.converged and self.spread < self.settle_std:
            self._settle_cnt += 1
        else:
            self._settle_cnt = 0
        self.settled = bool(self.converged and self._settle_cnt >= self.settle_hold)
        return entered_tracking

    def _enter_tracking(self, center_x, center_z):
        """Collapse a converged global posterior to its local spatial mode.

        Convergence is measured around the weighted mean, so the tracking cloud is
        the posterior support within ``tracking_radius_m`` of that same estimate.
        Systematic resampling restores the original particle count while permanently
        removing remote map aliases before normalized-frame tracking begins.
        """
        d = np.hypot(self.particles[:, 0] - float(center_x),
                     self.particles[:, 1] - float(center_z))
        keep = d <= self.tracking_radius_m
        if not np.any(keep):
            keep[int(np.argmax(self.weights))] = True
        particles = self.particles[keep]
        weights = self.weights[keep]
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            weights = np.full(len(particles), 1.0 / len(particles))
        else:
            weights = weights / total
        pos = (self.rng.rand() + np.arange(self.P)) / self.P
        idx = np.searchsorted(np.cumsum(weights), pos)
        idx = np.clip(idx, 0, len(particles) - 1)
        self.particles = particles[idx].copy()
        self.particles[:, 0] += self.rng.randn(self.P) * self.rough_xy
        self.particles[:, 1] += self.rng.randn(self.P) * self.rough_xy
        self.particles[:, 2] = wrap_pi(
            self.particles[:, 2] + self.rng.randn(self.P) * self.rough_yaw)
        self.logw = np.zeros(self.P)
        self.weights = np.full(self.P, 1.0 / self.P)
        self._cum_on = bool(self.cumulative_conf)
        self._cum_cnt = self.cum_init_cnt if self.cumulative_conf else 0

    def _resample(self):
        """Systematic (low-variance) resampling + roughening; weights reset."""
        pos = (self.rng.rand() + np.arange(self.P)) / self.P
        idx = np.searchsorted(np.cumsum(self.weights), pos)
        idx = np.clip(idx, 0, self.P - 1)
        self.particles = self.particles[idx]
        self.particles[:, 0] += self.rng.randn(self.P) * self.rough_xy
        self.particles[:, 1] += self.rng.randn(self.P) * self.rough_xy
        self.particles[:, 2] = wrap_pi(self.particles[:, 2]
                                       + self.rng.randn(self.P) * self.rough_yaw)
        # Augmented MCL: replace a slice with fresh uniform hypotheses so a lost
        # true pose can be rediscovered (skipped once converged -- there the cloud
        # is a small tracking blob and injection would only add noise).
        if self.inject_frac > 0.0 and not self.converged:
            n_inj = int(self.P * self.inject_frac)
            if n_inj > 0:
                self.particles[self.rng.choice(self.P, n_inj, replace=False)] = \
                    self._uniform_poses(n_inj)
        self.logw = np.zeros(self.P)
        self.weights = np.full(self.P, 1.0 / self.P)
        # A resample changes particle identities, but it must not make the next
        # single frame a 50% vote. Restore the configured post-lock inertia floor.
        self._cum_cnt = (self.cum_init_cnt
                         if self.converged and self.cumulative_conf else 0)

    def _uniform_poses(self, n):
        g = self._gmap
        fr, fc = np.nonzero(getattr(g, "explorable", g.free))
        i = self.rng.choice(len(fr), n, replace=True)
        out = np.zeros((n, 3))
        out[:, 0] = fc[i] * g.res + g.min_x + (self.rng.rand(n) - 0.5) * g.res
        out[:, 1] = fr[i] * g.res + g.min_z + (self.rng.rand(n) - 0.5) * g.res
        out[:, 2] = self.rng.uniform(-np.pi, np.pi, n)
        return out

    def _reglobalize(self):
        """Re-seed a fresh map-wide particle set (mirrors reset's global branch) and
        drop the (wrong) convergence, so the filter re-searches from scratch. Called
        by the kidnapped-robot recovery; motion gates must be satisfied again to re-commit."""
        g = self._gmap
        fr, fc = np.nonzero(getattr(g, "explorable", g.free))
        sel = np.arange(0, len(fr), self.pos_sparsity)
        gx = fc[sel] * g.res + g.min_x
        gz = fr[sel] * g.res + g.min_z
        if self.div_local:              # bound the re-search to a disk around the estimate
            near = np.hypot(gx - self.x, gz - self.z) <= self.div_local_radius
            if near.sum() >= self.angle_bin:
                gx, gz = gx[near], gz[near]
        yaws = np.arange(self.angle_bin) * (2 * np.pi / self.angle_bin) - np.pi
        n_pos = len(gx)
        if n_pos * self.angle_bin > self.max_particles:
            keep = max(1, self.max_particles // self.angle_bin)
            idx = self.rng.choice(n_pos, keep, replace=False)
            gx, gz, n_pos = gx[idx], gz[idx], keep
        P = np.zeros((n_pos * self.angle_bin, 3))
        P[:, 0] = np.repeat(gx, self.angle_bin)
        P[:, 1] = np.repeat(gz, self.angle_bin)
        P[:, 2] = np.tile(yaws, n_pos)
        self.particles = P
        self.P = len(P)
        self.logw = np.zeros(self.P)
        self.weights = np.full(self.P, 1.0 / self.P)
        self.converged = False
        self._cum_on = False
        self._cum_cnt = 0
        self.settled = False
        self._settle_cnt = 0
        self._hold = self._div_bad = 0
        self._accum_trans = self._accum_rot = self._total_trans = self._total_yaw = 0.0
        self._n_recover += 1

    # ------------------------------------------------------------------ #
    def update(self, obs=None, depth_key="depth", scan=None):
        # A caller may inject a precomputed scan; otherwise extract the requested depth
        # observation. The current sim and robot agents both use main-camera depth here.
        scan = self.extract_scan(obs, depth_key) if scan is None else np.asarray(scan)  # SINGLE frame, no buffer
        if self.accum_scan:                     # OPT-IN: fuse a windowed local submap instead
            scan = self._accumulate(scan)
        if self.lf is None or len(scan) < 10:
            self.x, self.z, self.yaw = self._estimate()
            return self.pose()
        if self.max_scan_points and len(scan) > self.max_scan_points:
            scan = scan[self.rng.choice(len(scan), self.max_scan_points, replace=False)]

        # frame informativeness -> temper this frame's weight + count informative
        # frames (EXTENSION; info_gating=False keeps beta=1 and the original path).
        info = self._frame_info(scan) if self.info_gating else 1.0
        self._last_info = info
        if info >= self.info_tau:
            self._n_info += 1
        beta = (self.info_beta_floor + (1.0 - self.info_beta_floor) * info
                if self.info_gating and self.info_temper else 1.0)

        # If the divergence-recovery check below will build the full score volume for THIS
        # scan, build it now and let _score_all read its per-particle cells from it -- the
        # same FFT serves both, instead of _score_all transforming its occupied subset again
        # (bit-identical). score_volume depends only on `scan`, so it is reused as-is by the
        # divergence branch. The predicate mirrors that branch but on the PRE-update
        # converged/step (converged only ever flips to True mid-update, never back to False
        # before the branch, so at worst the reuse is skipped on the single step it converges).
        div_vol = None
        if (self.divergence_recovery and self.converged and info >= self.info_tau
                and self._n_recover < self.div_max_recover
                and (self._steps + 1) % self.div_check_every == 0):
            _, div_vol = self.score_volume(scan)
        logL = self._score_all(scan, vol=div_vol)
        if self.cumulative_conf and self.converged:
            # TRACKING (reference SimpleParticles style): additively accumulate this
            # frame's NORMALIZED weight into a running average with growing inertia.
            #   weights <- (cnt * weights + w_frame) / (cnt + 1)
            # A bad frame's w_frame is bounded (sums to 1), so it can only nudge the
            # established belief by ~1/cnt instead of collapsing a particle (product).
            if not self._cum_on:                 # first frame after the lock: seed inertia
                self._cum_on = True
                self._cum_cnt = self.cum_init_cnt
            wf = np.exp(logL - logL.max())
            sf = wf.sum()
            if sf > 0:
                wf /= sf
            prior_count = self._cum_cnt
            self.weights = (self.weights * prior_count + wf) / (prior_count + 1)
            self.weights /= self.weights.sum()
            self._cum_cnt = prior_count + 1
        else:
            # GLOBAL SEARCH: cumulative likelihood = product of per-frame likelihoods
            # = sum of logs (reset at each resample). Bound each frame's loss relative
            # to its winner so one correlated/aliased view cannot permanently delete
            # otherwise viable global modes. beta<1 further tempers blank-wall frames.
            frame_logL = logL
            if self.global_loglik_cap is not None and self.global_loglik_cap > 0.0:
                frame_logL = logL - np.max(logL)
                finite = np.isfinite(frame_logL)
                frame_logL = np.where(
                    finite, np.maximum(frame_logL, -self.global_loglik_cap), frame_logL)
            self.logw += beta * frame_logL
            w = np.exp(self.logw - self.logw.max())
            self.weights = w / w.sum()

        self.x, self.z, self.yaw = self._estimate()
        entered_tracking = self._check_convergence()

        # inject more while still ambiguous (no informative frame yet) so an early
        # wrong collapse can recover; fall back to the base rate once informative.
        if self.info_gating:
            self.inject_frac = (self.ambig_inject_frac
                                if self._n_info < self.min_info_frames
                                else self._base_inject)

        # resample only after enough motion AND (optionally) at least one informative
        # frame, so a blank-wall corridor doesn't collapse the cloud prematurely.
        motion_ok = (self._accum_trans > self.resample_trans
                     or self._accum_rot > self.resample_rot)
        info_ok = (not self.info_gating or not self.require_info_resample
                   or self._n_info >= 1)
        if motion_ok and info_ok and not entered_tracking:
            self._resample()
            self._accum_trans = 0.0
            self._accum_rot = 0.0
        elif entered_tracking:
            # _enter_tracking already resampled the converged posterior.
            self._accum_trans = 0.0
            self._accum_rot = 0.0

        # kidnapped-robot recovery (EXTENSION): once converged, if the single-frame
        # MAP pose (global argmax) sits persistently FAR from the filter estimate on
        # informative frames, we locked onto a wrong (aliased) mode -> re-globalize.
        self._steps += 1
        if (self.divergence_recovery and self.converged and info >= self.info_tau
                and self._n_recover < self.div_max_recover
                and self._steps % self.div_check_every == 0):
            g = self._gmap
            # reuse the volume already built for _score_all this step (same scan) when present
            vol = div_vol if div_vol is not None else self.score_volume(scan)[1]
            mask = getattr(g, "explorable", g.free)
            vv = vol.copy()
            vv[:, ~mask] = -np.inf
            gmax = float(vv.max())               # best single-frame fit anywhere
            er = int(np.clip(round((self.z - g.min_z) / g.res), 0, g.H - 1))
            ec = int(np.clip(round((self.x - g.min_x) / g.res), 0, g.W - 1))
            step_yaw = 2 * np.pi / self.angle_bin
            eyb = int(round((self.yaw + np.pi) / step_yaw)) % self.angle_bin
            est_score = float(vol[eyb, er, ec])  # fit AT the current estimate
            self._div_bad = self._div_bad + 1 if (gmax - est_score) > self.div_gap else 0
            if self._div_bad >= self.div_hold:
                self._reglobalize()
        return self.pose()

    # ------------------------------------------------------------------ #
    def suggest_action(self, obs, depth_key="depth"):
        """DIRECTION-1 (active control only): while the current frame is too
        ambiguous to localize (e.g. facing a wall), suggest an action that seeks a
        more informative view -- turn toward the more OPEN side (more geometry to
        see) -- instead of committing to exploration. Returns an action id, or None
        to defer to the normal controller once the frame is informative / enough
        informative frames have accrued. Not used by the passive (oracle) evaluation.
        Toggle with active_seek (its own flag, so Direction 1 can be ablated
        without touching the info-gating logic).
        """
        if not self.active_seek:
            return None
        scan = self.extract_scan(obs, depth_key)
        if self._n_info >= self.min_info_frames or self._frame_info(scan) >= self.info_tau:
            return None                          # informative enough -> let controller run
        if len(scan) < 5:
            return self.A["turn_left"]           # nothing seen -> just rotate to look
        x, rng = scan[:, 0], np.hypot(scan[:, 0], scan[:, 1])
        left = rng[x < 0].mean() if (x < 0).any() else 0.0
        right = rng[x > 0].mean() if (x > 0).any() else 0.0
        return self.A["turn_left"] if left >= right else self.A["turn_right"]


def make_localizer(kind, **kw):
    """Build the one built-in localizer: the single-frame-scan particle filter
    (``particle_filter`` / ``single_frame``).

    Retired variants (scan_match, accumulate-grid particle, dead_reckon) were moved
    OUT of the package to ``legacy/localizers.py`` and are intentionally not imported
    here, so the main pipeline never depends on them. To use one, copy it back into
    ``navigation/`` or import it explicitly from ``legacy.localizers``."""
    if kind in ("particle_filter", "single_frame", "pf", "mcl", "smcl"):
        return ParticleFilterLocalizer(**kw)
    raise ValueError(
        f"unknown localizer '{kind}'. The only built-in localizer is "
        f"'particle_filter' (alias 'single_frame'); retired variants "
        f"(scan_match / particle / dead_reckon) now live in legacy/localizers.py.")
