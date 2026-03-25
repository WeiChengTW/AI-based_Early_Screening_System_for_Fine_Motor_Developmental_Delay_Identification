import cv2, glob, os, math, numpy as np
from skimage.morphology import skeletonize
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class CrossScorer:
    def __init__(
        self,
        *,
        cm_per_pixel=None,  # None=用像素；填 0.02079 之類就用公分
        angle_min=70.0,
        angle_max=110.0,
        max_spread_cm=0.6,  # 只有在 cm_per_pixel 有值時才會使用
        out_dir=None,
        output_jpg_quality=95,
    ):
        if out_dir is None:
            out_dir = str(BASE_DIR / "output")

        self.cm_per_pixel = cm_per_pixel
        self.ANGLE_MIN = angle_min
        self.ANGLE_MAX = angle_max
        self.MAX_SPREAD_CM = max_spread_cm
        self.OUT_DIR = out_dir
        self.OUTPUT_JPG_QUALITY = output_jpg_quality
        os.makedirs(self.OUT_DIR, exist_ok=True)

    # ---------- UI ----------
    def put_text_top_left(
        self,
        img,
        lines,
        org=(10, 120),  # 稍微往下移一點，避免貼邊
        k=0.0015,  # 稍微調小字體係數
    ):
        H, W = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 根據圖片高度自動計算字體大小與粗細
        fs = max(0.5, H * k) * 0.5
        thick = max(1, int(fs * 2))
        vgap = int(60 * fs)  # 行距

        y = org[1]
        for t in lines:
            # 繪製文字描邊（黑色），增加可視度
            cv2.putText(
                img, t, (org[0], y), font, fs, (0, 0, 0), thick + 2, cv2.LINE_AA
            )
            # 繪製主文字（白色）
            cv2.putText(
                img, t, (org[0], y), font, fs, (255, 255, 255), thick, cv2.LINE_AA
            )
            y += vgap

    # ---------- 幾何小工具 ----------
    @staticmethod
    def calculate_distance_numpy(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    @staticmethod
    def angle_between(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
        return math.degrees(math.acos(c))  # 0~180°

    @staticmethod
    def endpoints_junctions_center(S):
        S = (S > 0).astype(np.uint8)
        if not S.any():
            return [], [], None
        k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
        deg = cv2.filter2D(S, -1, k, borderType=cv2.BORDER_CONSTANT)
        ey, ex = np.where((S == 1) & (deg == 1))
        jy, jx = np.where((S == 1) & (deg >= 3))
        ys, xs = np.where(S == 1)
        C = (float(xs.mean()), float(ys.mean()))
        ends = [(int(x), int(y)) for y, x in zip(ey, ex)]
        junc = [(int(x), int(y)) for y, x in zip(jy, jx)]
        return ends, junc, C

    @staticmethod
    def points_near_segment(pts, A, B, max_dist=3.0, margin=10.0):
        A, B = np.asarray(A, float), np.asarray(B, float)
        v = B - A
        vv = v.dot(v)
        if vv < 1e-6:
            return np.empty((0, 2), float)
        t = ((pts - A) @ v) / vv
        proj = A + t[:, None] * v
        dist = np.hypot(*(pts - proj).T)
        tmin, tmax = -margin / (math.sqrt(vv) + 1e-9), 1 + margin / (
            math.sqrt(vv) + 1e-9
        )
        return pts[(dist <= max_dist) & (t >= tmin) & (t <= tmax)]

    @staticmethod
    def fit_line_pca(pts):
        if pts.shape[0] < 5:
            return None, None
        m = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - m, full_matrices=False)
        d = Vt[0]
        n = np.linalg.norm(d)
        return (m, d / n) if n >= 1e-9 else (None, None)

    @staticmethod
    def line_intersection(P0, d0, P1, d1):
        c = float(np.clip(d0.dot(d1), -1, 1))
        ang = math.degrees(math.acos(abs(c)))
        if ang < 10:
            return None
        A = np.stack([d0, -d1], 1)
        b = P1 - P0
        try:
            s, _ = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        return (float(P0[0] + s * d0[0]), float(P0[1] + s * d0[1]))

    # ---------- 骨架找十字 ----------
    def robust_cross_from_skeleton(self, skeleton):
        S = (skeleton > 0).astype(np.uint8) * 255  # 一開始可能只抓到非圖形區塊的原因
        ends, junc, C = self.endpoints_junctions_center(S)
        if len(ends) < 4:
            if junc:
                P = min(junc, key=lambda p: (p[0] - C[0]) ** 2 + (p[1] - C[1]) ** 2)
            else:
                return False, "need 4 endpoints", None, None, None, None, ends
            return False, "too few endpoints", P, None, None, None, ends

        # 取離中心最遠的四端點
        ends = sorted(
            ends, key=lambda p: (p[0] - C[0]) ** 2 + (p[1] - C[1]) ** 2, reverse=True
        )[:4]
        ys, xs = np.where(S > 0)
        sk_pts = np.stack([xs, ys], 1).astype(float)
        pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]

        best = None
        best_score = -1e18
        for (a, b), (c, d) in pairings:
            A, B = ends[a], ends[b]
            C2, D = ends[c], ends[d]
            pts1 = self.points_near_segment(sk_pts, A, B)
            pts2 = self.points_near_segment(sk_pts, C2, D)
            P0, d0 = self.fit_line_pca(pts1)
            P1, d1 = self.fit_line_pca(pts2)
            if P0 is None:
                v0 = np.array([B[0] - A[0], B[1] - A[1]], float)
                v0 /= np.linalg.norm(v0) + 1e-9
                P0, d0 = np.array(A, float), v0
            if P1 is None:
                v1 = np.array([D[0] - C2[0], D[1] - C2[1]], float)
                v1 /= np.linalg.norm(v1) + 1e-9
                P1, d1 = np.array(C2, float), v1
            Pint = self.line_intersection(P0, d0, P1, d1)
            if Pint is None:
                continue
            theta = self.angle_between(d0, d1)
            n1p, n2p = len(pts1), len(pts2)
            c90 = -abs((theta or 0) - 90)
            cC = -math.hypot(Pint[0] - C[0], Pint[1] - C[1])
            pair_score = 2.0 * c90 + 0.01 * (n1p + n2p) + 0.002 * cC
            if pair_score > best_score:
                best_score = pair_score
                best = (Pint, (A, B), (C2, D), theta)

        if best is None:
            if junc:
                P = min(junc, key=lambda p: (p[0] - C[0]) ** 2 + (p[1] - C[1]) ** 2)
                return False, "two lines do not intersect", P, None, None, None, ends
            return False, "two lines do not intersect", None, None, None, None, ends

        P, pair1, pair2, theta = best
        return True, "", P, pair1, pair2, theta, ends

    # ---------- 單張處理 ----------
    def score_image(self, img_path):
        result = {
            "path": img_path,
            "score": 0,
            "reason": "init",
            "theta_deg": None,
            "arms_px": None,
            "arms_cm": None,
            "spread_px": None,
            "spread_cm": None,
        }

        # ---- 前處理 ----
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            result["reason"] = "cannot read image"
            return result, None, None, None

        blur = cv2.GaussianBlur(img, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # cv2.imshow('ori', img)
        # cv2.imshow('binary', binary)

        # cv2.imshow 顯示灰階時是把 0 視為黑、255 視為白
        skel_bool = skeletonize(binary > 0)  # True/False
        skel = skel_bool.astype(np.uint8) * 255  # 0/255，便於顯示
        # cv2.imshow('bin', binary)
        # cv2.imshow('skel', skel)
        # cv2.waitKey(0)

        ok, reason, P, p1, p2, theta, ends = self.robust_cross_from_skeleton(skel)

        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        skel_vis = cv2.dilate(
            skel, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1
        )
        overlay = vis.copy()
        overlay[skel_vis > 0] = [0, 0, 255]
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        score = 0
        shown_reason = "no intersection"
        dists_vals, spread_val = None, None

        if ok and P is not None and theta is not None and p1 and p2:
            pts = [p1[0], p1[1], p2[0], p2[1]]
            d_px = [
                self.calculate_distance_numpy(P, (x, y)) for (x, y) in pts
            ]  # 交點→四端點

            # 單位處理
            if self.cm_per_pixel is None:
                result["arms_px"] = [float(d) for d in d_px]
                result["spread_px"] = float(max(d_px) - min(d_px)) if d_px else None
                dists_vals = result["arms_px"]
                spread_val = result["spread_px"]
                spread_ok = True  # 不用 cm 門檻
            else:
                d_cm = [d * self.cm_per_pixel for d in d_px]
                result["arms_px"] = [float(d) for d in d_px]
                result["arms_cm"] = [float(d) for d in d_cm]
                result["spread_px"] = float(max(d_px) - min(d_px)) if d_px else None
                result["spread_cm"] = float(max(d_cm) - min(d_cm)) if d_cm else None
                dists_vals = result["arms_cm"]
                spread_val = result["spread_cm"]
                spread_ok = spread_val is not None and spread_val <= self.MAX_SPREAD_CM

            angle_ok = self.ANGLE_MIN <= theta <= self.ANGLE_MAX

            if angle_ok and spread_ok:
                score = 2
                shown_reason = (
                    "angle ok"
                    if self.cm_per_pixel is None
                    else f"angle ok & spread<={self.MAX_SPREAD_CM:.2f}cm"
                )
            else:
                score = 1
                fails = []
                if not angle_ok:
                    fails.append(
                        f"angle {theta:.2f} not in [{self.ANGLE_MIN:.0f},{self.ANGLE_MAX:.0f}]"
                    )
                if self.cm_per_pixel is not None and not spread_ok:
                    fails.append(
                        f"spread {spread_val:.2f}cm > {self.MAX_SPREAD_CM:.2f}cm"
                    )
                shown_reason = "; ".join(fails) if fails else "partial"

            # 視覺化線段與交點
            cv2.line(vis, p1[0], p1[1], (0, 255, 0), 2)
            cv2.line(vis, p2[0], p2[1], (255, 0, 0), 2)
            cv2.circle(vis, (int(round(P[0])), int(round(P[1]))), 5, (0, 0, 255), -1)

            result["theta_deg"] = float(theta)

        else:
            score = 0
            shown_reason = reason or "no intersection"

        # 標出端點（若有）
        for pt in ends[:4] if ends else []:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

        # 面板
        theta_str = (
            "-" if result["theta_deg"] is None else f'{result["theta_deg"]:.2f} deg'
        )
        arms_px_str = (
            "-"
            if result["arms_px"] is None
            else ", ".join(f"{d:.2f}" for d in result["arms_px"])
        )
        panel = [
            f"Score: {score}",
            f"Reason: {shown_reason}",
            f"Theta: {theta_str}",
            f"Arms(px): {arms_px_str}",
        ]
        if self.cm_per_pixel is not None:
            arms_cm_str = (
                "-"
                if result["arms_cm"] is None
                else ", ".join(f"{d:.2f}" for d in result["arms_cm"])
            )
            spread_cm_str = (
                "-" if result["spread_cm"] is None else f"{result['spread_cm']:.2f} cm"
            )
            panel.append(f"Arms(cm): {arms_cm_str}")
            panel.append(f"Spread: {spread_cm_str}")
        else:
            spread_px_str = (
                "-" if result["spread_px"] is None else f"{result['spread_px']:.2f} px"
            )
            panel.append(f"Spread: {spread_px_str}")

        self.put_text_top_left(vis, panel)

        # 填結果
        result["score"] = int(score)
        result["reason"] = shown_reason

        # 檔名
        base = os.path.basename(img_path)
        img_path = os.path.join(self.OUT_DIR, f"img_path_{base}")
        bin_path = os.path.join(self.OUT_DIR, f"binary_{base}")
        skel_path = os.path.join(self.OUT_DIR, f"skeleton_{base}")
        vis_path = os.path.join(self.OUT_DIR, f"processed_{base}")

        # cv2.imshow('vis', vis)
        # cv2.waitKey(0)
        # 輸出
        cv2.imwrite(img_path, img)
        cv2.imwrite(bin_path, binary)
        cv2.imwrite(skel_path, skel)
        cv2.imwrite(vis_path, vis, [cv2.IMWRITE_JPEG_QUALITY, self.OUTPUT_JPG_QUALITY])

        return result, vis, bin_path, skel_path, vis_path

    # ---------- 批次處理 ----------
    def score_folder(self, image_folder, pattern):
        paths = sorted(glob.glob(os.path.join(image_folder, pattern)))
        print("找到圖片檔案:", paths)
        results = []
        for p in paths:
            print("\n處理圖片:", os.path.basename(p))
            res, b, s, v = self.score_image(p)
            print("Saved:", v)
            results.append(res)
        return results


if __name__ == "__main__":
    # 1) 像素版（不做 cm 換算）
    # scorer = CrossScorer(cm_per_pixel=None, angle_min=70.0, angle_max=110.0, max_spread_cm=0.6)

    # 2) 若要公分版，改成：
    scorer = CrossScorer(
        cm_per_pixel=0.02079,
        angle_min=70.0,
        angle_max=110.0,
        max_spread_cm=0.6,
        out_dir=str(BASE_DIR / "output"),
    )

    # results = scorer.score_folder("image", r"realtest")#"img*.jpg"
    results = scorer.score_image(r"2.jpg")
    # score = results['score']
    print("所有結果:", results)
