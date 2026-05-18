import numpy as np
from typing import Dict, Any, List


def _major_axis_direction(box: np.ndarray) -> np.ndarray:
    """
    (tl, tr, br, bl) の順に並んだ box から，長辺方向の単位ベクトルを返す。
    中心軸と平行な向きになる。
    """
    box = np.asarray(box, dtype=np.float32).reshape(4, 2)
    edges = [
        box[1] - box[0],  # top
        box[2] - box[1],  # right
        box[3] - box[2],  # bottom
        box[0] - box[3],  # left
    ]
    lens = [np.linalg.norm(e) for e in edges]
    idx = int(np.argmax(lens))
    v = edges[idx]
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([0.0, -1.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _long_edges_left_right(box: np.ndarray):
    """
    (tl, tr, br, bl) の box から、
    長辺2本を取り、そのうち
      - 左側の長辺 (left_edge: (p0, p1))
      - 右側の長辺 (right_edge: (p0, p1))
    を返す。
    """
    b = np.asarray(box, dtype=np.float32).reshape(4, 2)
    edges = [
        (b[0], b[1]),  # top
        (b[1], b[2]),  # right
        (b[2], b[3]),  # bottom
        (b[3], b[0]),  # left
    ]
    lens = [np.linalg.norm(p1 - p0) for (p0, p1) in edges]
    max_len = max(lens)

    eps = max_len * 1e-3
    long_ids = [k for k, L in enumerate(lens) if L >= max_len - eps]
    if len(long_ids) < 2:
        long_ids = list(np.argsort(lens)[-2:])

    candidates = []
    for k in long_ids:
        p0, p1 = edges[k]
        mean_x = float((p0[0] + p1[0]) * 0.5)
        candidates.append((mean_x, p0, p1))

    candidates.sort(key=lambda t: t[0])  # x 小さい方が左
    _, left_p0, left_p1 = candidates[0]
    _, right_p0, right_p1 = candidates[-1]
    return (left_p0, left_p1), (right_p0, right_p1)


def run_step7_compute_guide_line(
    ctx: Dict[str, Any],
    line_len: float = 500.0,
) -> Dict[str, Any]:
    """
    STEP 7: ガイドライン（赤線）の始点 p0 / 終点 p1 を決める。

    ・x の求め方だけ、ノートの二次方程式ベースに変更。
    ・p0 の決め方は「左半分/右半分」「どちらの本が傾いているか」で
      4パターンに分岐。
    """
    rects: List[np.ndarray] = ctx.get("rects", [])
    axis_index = ctx.get("axis_index")
    base_point = ctx.get("base_point")
    midpoint = ctx.get("midpoint")

    ctx["line_p0"] = None
    ctx["line_p1"] = None
    ctx["sub_line_p0"] = None
    ctx["sub_line_p1"] = None

    if axis_index is None or base_point is None or midpoint is None:
        return ctx
    if not (0 <= axis_index < len(rects)):
        return ctx

    axis_rect = rects[axis_index]

    # ガイド線（赤線）の向き
    axis_dir = _major_axis_direction(axis_rect)  # (2,) unit
    base_pt = np.asarray(base_point, dtype=np.float32)
    mid = np.asarray(midpoint, dtype=np.float32)

    # 中点方向に向くように符号を揃える
    v_mid = mid - base_pt
    if np.dot(axis_dir, v_mid) < 0.0:
        axis_dir = -axis_dir

    # ペア情報
    pair = ctx.get("pair_indices")      # (i, j)
    centers = ctx.get("centers")        # shape: (N, 2)
    is_right_half = ctx.get("is_right_half")

    # どちらが傾いているか（無ければ False）
    left_is_tilted  = bool(ctx.get("left_is_tilted", False))
    right_is_tilted = bool(ctx.get("right_is_tilted", False))

    if pair is None or centers is None or is_right_half is None:
        p0 = base_pt
        p1 = base_pt + axis_dir * float(line_len)
        ctx["line_p0"] = p0
        ctx["line_p1"] = p1
        return ctx

    centers = np.asarray(centers, dtype=np.float32)
    i, j = pair

    # x 座標で左本 / 右本
    cx_i = float(centers[i][0])
    cx_j = float(centers[j][0])
    if cx_i <= cx_j:
        left_idx, right_idx = i, j
    else:
        left_idx, right_idx = j, i

    # is_right_half:
    #   True  → 右半分にペア → 信頼できるのは左本（右側がギャップ）
    #   False → 左半分にペア → 信頼できるのは右本（左側がギャップ）
    if is_right_half:
        target_idx = left_idx
        want_right_side = True
    else:
        target_idx = right_idx
        want_right_side = False

    if not (0 <= target_idx < len(rects)):
        return ctx

    # 対象本の側面
    box_tgt = rects[target_idx]
    (left_edge, right_edge) = _long_edges_left_right(box_tgt)
    if want_right_side:
        s0, s1 = right_edge
    else:
        s0, s1 = left_edge

    side_p0 = np.asarray(s0, dtype=np.float32)
    side_p1 = np.asarray(s1, dtype=np.float32)

    # 上端 / 下端（y 小さい方が上）
    if side_p0[1] <= side_p1[1]:
        sub0 = side_p0
        sub1 = side_p1
    else:
        sub0 = side_p1
        sub1 = side_p0

    # sub0 → sub1 が「下」方向
    side_dir = sub1 - sub0
    n_side = float(np.linalg.norm(side_dir))
    if n_side < 1e-6:
        p0 = base_pt
        p1 = base_pt + axis_dir * float(line_len)
        ctx["line_p0"] = p0
        ctx["line_p1"] = p1
        ctx["sub_line_p0"] = side_p0
        ctx["sub_line_p1"] = side_p1
        return ctx
    side_dir = side_dir / n_side

    # ===== x, y を二次方程式から求める部分 =====
    d = float(line_len)
    cos_theta = float(axis_dir[0])
    cos_phi   = float(side_dir[0])

    def solve_x_for_case(ct: float, cp: float, c_ref: float):
        """
        y = d * ct - (d - x) * cp
        (d - x)^2 = y^2 + d^2 - 2 d y c_ref
        → A x^2 + B x + C = 0 を解いて (x, y) を返す
        """
        A = 1.0 - cp * cp
        B = 2.0 * d * (cp * cp + cp * c_ref - cp * ct - 1.0)
        C = d * d * (-cp * cp - 2.0 * cp * c_ref + 2.0 * cp * ct + 2.0 * c_ref * ct - ct * ct)

        if abs(A) < 1e-8:
            if abs(B) < 1e-8:
                return None, None
            x_lin = -C / B
            if 0.0 <= x_lin <= d:
                y_lin = d * ct - (d - x_lin) * cp
                return float(x_lin), float(y_lin)
            return None, None

        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            print("[WARN] solve_x_for_case: 判別式が負です。")
            return None, None
        sqrt_disc = float(np.sqrt(max(disc, 0.0)))

        x1 = (-B + sqrt_disc) / (2.0 * A)
        x2 = (-B - sqrt_disc) / (2.0 * A)

        cand = []
        for xv in (x1, x2):
            yv = d * ct - (d - xv) * cp
            cand.append((float(xv), float(yv)))

        if not cand:
            print("[WARN] solve_x_for_case: 有効な解がありません。")
            return None, None
        print("[DEBUG] solve_x_for_case: candidates =", cand)
        cand.sort(key=lambda t: t[0])   # x が小さい解を優先
        return cand[0]

    # ---- ここからが「式の切り替えロジック」修正部分 ----
    # 右の本が傾いているか / 左の本が傾いているか で式を決める
    if right_is_tilted and not left_is_tilted:
        # 「右の書籍が傾いている」ケース
        #   y = d*cosθ - (d-x)*cosφ
        #   (d-x)^2 = y^2 + d^2 - 2*d*y*cosθ
        ct, cp, c_ref = cos_theta, cos_phi, cos_theta

    elif left_is_tilted and not right_is_tilted:
        # 「左の書籍が傾いている」ケース
        #   y = d*cos(180-φ) - (d-x)*cos(180-θ)
        #   (d-x)^2 = y^2 + d^2 - 2*d*y*cos(180-φ)
        #   → cos(180-α) = -cosα を使う
        ct, cp, c_ref = -cos_phi, -cos_theta, -cos_phi

    else:
        # フォールバック（両方 False / 両方 True など）
        # 以前の符号判定ロジックを保険として残す
        if cos_theta * cos_phi >= 0.0:
            ct, cp, c_ref = cos_theta, cos_phi, cos_theta
        else:
            ct, cp, c_ref = -cos_phi, -cos_theta, -cos_phi

    x, y_val = solve_x_for_case(ct, cp, c_ref)

    if x is None or y_val is None:
        # どうしても解が出ないときの最終保険
        x = d
        y_val = 0.0


    side_line0 = sub0  # 上端
    side_line1 = sub1  # 下端

    # ===== p0 の位置（あなたの指定ロジック） =====
    if not is_right_half:
        # ペアが左半分
        if right_is_tilted:
            p0 = side_line0 + axis_dir * x
            print("[STEP7] left half, right tilted")
        else:
            p0 = side_line0 - side_dir * x
            print("[STEP7] left half, left tilted")    
    else:
        # ペアが右半分
        if left_is_tilted:
            p0 = side_line0 - axis_dir * x
            print("[STEP7] right half, left tilted")
        else:
            p0 = side_line0 + side_dir * x
            print("[STEP7] right half, right tilted")
    p1 = p0 + axis_dir * d

    # デバッグ用出力
    print("[STEP7] x =", float(x))
    print("[STEP7] y =", float(y_val))
    print("[STEP7] p0 =", p0)
    print("[STEP7] p1 =", p1)
    print("[STEP7] side_line0 =", side_line0)
    print("[STEP7] side_line1 =", side_line1)
        

    ctx["line_p0"] = p0.astype(np.float32)
    ctx["line_p1"] = p1.astype(np.float32)
    ctx["sub_line_p0"] = side_line0.astype(np.float32)
    ctx["sub_line_p1"] = side_line1.astype(np.float32)
    ctx["geom_y_val"] = float(y_val)
    ctx["right_is_tilted"] = right_is_tilted
    ctx["right_half"]= is_right_half

    return ctx
