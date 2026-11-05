import cv2 as cv
import numpy as np
import os
import pandas as pd
import math

# ==========================================
# 0. 全局视觉设置
# ==========================================
LW = 15           # 线条宽度 (Line Width)
FONT_SCALE = 5    # e_size 文字大小
FONT_THICK = 10   # 文字粗细

# ==========================================
# 1. 核心算法工具
# ==========================================

def get_fractal_dimension(gray_img, threshold=0.9):
    """计算分形维数"""
    Z = (gray_img < (threshold * 255))
    p = min(Z.shape)
    if p < 4: return np.nan
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], size), axis=0),
                               np.arange(0, Z.shape[1], size), axis=1)
        counts.append(len(np.where((S > 0) & (S < size*size))[0]))
    if not counts or len(counts) < 2: return np.nan
    valid = np.array(counts) > 0
    if not np.any(valid): return np.nan
    coeffs = np.polyfit(np.log(sizes[valid]), np.log(np.array(counts)[valid]), 1)
    return -coeffs[0]

def get_glcm_features(gray_img, mask):
    """计算GLCM纹理特征"""
    laplacian = cv.Laplacian(gray_img, cv.CV_64F)
    laplacian_masked = laplacian[mask > 0]
    tex_contrast = np.std(laplacian_masked) if len(laplacian_masked) > 0 else 0
    hist = cv.calcHist([gray_img], [0], mask, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-6)
    tex_entropy = -np.sum(hist * np.log(hist + 1e-6))
    tex_asm = np.sum(hist ** 2)
    return tex_contrast, tex_entropy, tex_asm

def analyze_gaps_internal(mask_filled, thresh):
    """Gap分析逻辑"""
    mask_gap = (mask_filled > 0) & (thresh == 0)
    mask_gap = mask_gap.astype(np.uint8) * 255
    gap_area = np.sum(mask_gap > 0)
    area_total = np.sum(mask_filled > 0)
    gap_ratio = gap_area / area_total if area_total > 0 else 0
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask_gap, connectivity=8)
    gap_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area > 5: gap_areas.append(area)
    num_gaps = len(gap_areas)
    mean_gap_area = np.mean(gap_areas) if gap_areas else 0
    max_gap_area = np.max(gap_areas) if gap_areas else 0
    return {
        'num_gaps': num_gaps, 'gap_ratio': gap_ratio,
        'mean_gap_area': mean_gap_area, 'max_gap_area': max_gap_area
    }, mask_gap

# ==========================================
# 2. 辅助工具：动态画布重构
# ==========================================

def create_centered_canvas(img, cnt, hull):
    """创建一个足够大的正方形画布，并将物体居中"""
    h, w = img.shape[:2]
    (x, y), radius = cv.minEnclosingCircle(cnt)
    new_size = int(max(h, w, radius * 2) * 1.5)
    canvas = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    x_old, y_old, w_old, h_old = cv.boundingRect(cnt)
    offset_x = (new_size - w_old) // 2 - x_old
    offset_y = (new_size - h_old) // 2 - y_old
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    img_centered = cv.warpAffine(img, M, (new_size, new_size))
    cnt_centered = cnt + np.array([offset_x, offset_y])
    hull_centered = hull + np.array([offset_x, offset_y])
    return img_centered, cnt_centered, hull_centered

# ==========================================
# 3. 可视化绘图模块 (v7 完美版)
# ==========================================

def save_vis_morphology(save_path, img, cnt, hull):
    """a. Morphology"""
    vis, cnt_new, hull_new = create_centered_canvas(img, cnt, hull)
    
    (x, y), radius = cv.minEnclosingCircle(cnt_new)
    center = (int(x), int(y))
    radius = int(radius)
    cv.circle(vis, center, radius, (0, 255, 0), LW)

    rect = cv.minAreaRect(cnt_new)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(vis, [box], 0, (255, 0, 0), LW)

    cv.drawContours(vis, [hull_new], -1, (255, 0, 255), LW)
    cv.drawContours(vis, [cnt_new], -1, (0, 140, 255), LW)

    x_b, y_b, w_b, h_b = cv.boundingRect(cnt_new)
    cx = x_b + w_b // 2
    img_h = vis.shape[0]
    cv.arrowedLine(vis, (cx, img_h - 50), (cx, 50), (0, 0, 255), LW, tipLength=0.03)

    cv.imwrite(save_path, vis)

def save_vis_gaps_binary(save_path, mask_gaps):
    """Gaps Mask"""
    cv.imwrite(save_path, mask_gaps)

def save_vis_structure_radial(save_path, img, cnt):
    """b. Structure: 修复版 (独立逻辑)"""
    try:
        x, y, w, h = cv.boundingRect(cnt)
        cx_old = x + w // 2
        cy_old = y + h // 2
        pts = cnt.reshape(-1, 2)
        dists = np.linalg.norm(pts - np.array([cx_old, cy_old]), axis=1)
        max_dist = np.max(dists) if len(dists) > 0 else max(w, h)/2

        canvas_size = int(max_dist * 2.4) + 100
        vis = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        cx_new = canvas_size // 2
        cy_new = canvas_size // 2
        shift_x = cx_new - cx_old
        shift_y = cy_new - cy_old
        cnt_shift = cnt + np.array([shift_x, shift_y])

        colors = [(130, 0, 75), (255, 0, 0), (255, 255, 0), (0, 255, 0),
                  (0, 255, 255), (0, 165, 255), (0, 0, 255), (255, 0, 255)]
        
        num_circles = 8
        step = max_dist / num_circles if max_dist > 0 else 10
        for i in range(1, num_circles + 2):
            radius = int(i * step)
            color = colors[(i-1) % len(colors)]
            cv.circle(vis, (cx_new, cy_new), radius, color, LW)

        cv.drawContours(vis, [cnt_shift], -1, (255, 255, 255), LW)
        cv.imwrite(save_path, vis)
    except Exception as e:
        print(f"[Error] Failed to create structure map: {e}")

def save_vis_color(save_path, img):
    """c. Color"""
    b, g, r = cv.split(img)
    zeros = np.zeros_like(b)
    img_R = cv.merge([zeros, zeros, r])
    img_G = cv.merge([zeros, g, zeros])
    img_B = cv.merge([b, zeros, zeros])
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    img_H = cv.cvtColor(h, cv.COLOR_GRAY2BGR)
    img_S = cv.cvtColor(s, cv.COLOR_GRAY2BGR)
    img_V = cv.cvtColor(v, cv.COLOR_GRAY2BGR)
    row1 = np.hstack([img_R, img_G, img_B])
    row2 = np.hstack([img_H, img_S, img_V])
    grid = np.vstack([row1, row2])
    if grid.shape[0] > 2000: grid = cv.resize(grid, (0, 0), fx=0.3, fy=0.3)
    cv.imwrite(save_path, grid)

def save_vis_texture_cluster(save_path, img, mask):
    """d. Texture"""
    scale_percent = 25 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img_small = cv.resize(img, (width, height))
    mask_small = cv.resize(mask, (width, height), interpolation=cv.INTER_NEAREST)
    masked_img = cv.bitwise_and(img_small, img_small, mask=mask_small)
    data = masked_img.reshape((-1, 3)).astype(np.float32)
    valid_pixels = data[np.all(data != 0, axis=1)]
    if len(valid_pixels) > 0:
        K = 8
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv.kmeans(valid_pixels, K, None, criteria, 5, cv.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        color_map = np.random.randint(50, 255, (K, 3), dtype=np.uint8)
        res_small = np.zeros_like(data, dtype=np.uint8)
        valid_indices = np.where(np.all(data != 0, axis=1))[0]
        res_small[valid_indices] = color_map[label.flatten()]
        vis_small = res_small.reshape(img_small.shape)
        vis = cv.resize(vis_small, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)
    else:
        vis = np.zeros_like(img)
    cv.imwrite(save_path, vis)

def save_vis_size_style(save_path, img, cnt, hull):
    """e. Size: 修正布局 (长轴 + 居中文字)"""
    vis = img.copy()
    h, w = vis.shape[:2]
    
    cv.drawContours(vis, [cnt], -1, (255, 0, 255), LW)
    
    center_x, center_y = w // 2, h // 2
    if len(cnt) >= 5:
        (xc, yc), (ma, ma), angle = cv.fitEllipse(cnt)
        length = max(h, w) * 2
        
        def get_intersection(test_angle):
            rad = np.deg2rad(test_angle)
            dx = math.cos(rad)
            dy = math.sin(rad)
            mask_line = np.zeros((h, w), dtype=np.uint8)
            pt1 = (int(xc - length * dx), int(yc - length * dy))
            pt2 = (int(xc + length * dx), int(yc + length * dy))
            cv.line(mask_line, pt1, pt2, 255, 5)
            mask_hull = np.zeros((h, w), dtype=np.uint8)
            cv.drawContours(mask_hull, [hull], -1, 255, -1)
            inter = cv.bitwise_and(mask_hull, mask_line)
            pts = cv.findNonZero(inter)
            if pts is None: return 0, None, None
            p_data = pts.reshape(-1, 2)
            projs = p_data[:, 0] * dx + p_data[:, 1] * dy
            dist = np.max(projs) - np.min(projs)
            return dist, tuple(p_data[np.argmin(projs)]), tuple(p_data[np.argmax(projs)])

        l1, s1, e1 = get_intersection(angle)
        l2, s2, e2 = get_intersection(angle + 90)
        
        if l1 > l2 and s1 and e1:
            cv.line(vis, s1, e1, (255, 255, 0), LW)
            center_x, center_y = int((s1[0] + e1[0])/2), int((s1[1] + e1[1])/2)
        elif s2 and e2:
            cv.line(vis, s2, e2, (255, 255, 0), LW)
            center_x, center_y = int((s2[0] + e2[0])/2), int((s2[1] + e2[1])/2)

    x_rect, y_rect, w_rect, h_rect = cv.boundingRect(cnt)
    def put_text_safe(text, x, y, color):
        cv.putText(vis, text, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICK)

    put_text_safe("area", x_rect + w_rect * 0.1, y_rect + h_rect * 0.2, (0, 255, 0))
    diam_size = cv.getTextSize("diameter", cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)[0]
    put_text_safe("diameter", center_x - diam_size[0] // 2, center_y, (255, 255, 0))
    perim_size = cv.getTextSize("perimeter", cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)[0]
    put_text_safe("perimeter", center_x - perim_size[0] // 2, y_rect + h_rect - (h_rect * 0.1), (255, 0, 255))
    
    cv.imwrite(save_path, vis)

# ==========================================
# 4. 主程序 (集成 Pro 参数 + 数据筛选)
# ==========================================

if __name__ == "__main__":
    IMG_DIR = "./images/"
    RAW_DATA_PATH = "1_raw_data.txt"
    OUTPUT_CSV = "cauliflower_analysis_final2.csv"
    VIS_DIR = "./visual_outputs_final2/"
    
    if not os.path.exists(VIS_DIR): os.makedirs(VIS_DIR)

    # 1. 读取数据 (仅保留 GR 和 DD)
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH, sep='\t')
        
        # ID 清洗
        df_raw['match_id'] = df_raw['image_id'].astype(str)
        df_raw['match_id'] = df_raw['match_id'].str.replace(r'\.jpg|\.png|\.jpeg', '', regex=True, case=False)
        df_raw['match_id'] = df_raw['match_id'].str.strip().str.lstrip('0')
        
        # 【关键修改】数据筛选：只保留 match_id, GR, DD
        cols_to_keep = ['match_id', 'GR', 'DD']
        # 检查列是否存在，防止报错
        cols_exist = [c for c in cols_to_keep if c in df_raw.columns]
        
        if 'GR' in cols_exist and 'DD' in cols_exist:
            df_raw = df_raw[cols_exist] # 执行筛选
            # 剔除空值
            df_raw_clean = df_raw.dropna(subset=['GR', 'DD'])
            print(f"[信息] 原始数据加载成功。保留列: {cols_exist}。有效样本: {len(df_raw_clean)}")
        else:
            print("[错误] 1_raw_data.txt 缺失 GR 或 DD 列。")
            exit()
            
    except Exception as e:
        print(f"[错误] 无法读取 {RAW_DATA_PATH}: {e}")
        exit()

    records = []
    print(f"--- 开始处理图像 (输出格式: JPG) ---")
    print(f"--- 输出目录: {VIS_DIR} ---")

    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    total_imgs = len(img_files)

    for idx, fname in enumerate(img_files):
        if (idx + 1) % 10 == 0:
            print(f"    处理进度: {idx + 1}/{total_imgs}")

        name_no_ext = os.path.splitext(fname)[0]
        base_match_id = name_no_ext.lstrip('0')
        if base_match_id == '': base_match_id = '0'
        
        base_name = name_no_ext
        img_path = os.path.join(IMG_DIR, fname)
        img = cv.imread(img_path)
        if img is None: continue
        
        # 图像处理
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = max(contours, key=cv.contourArea)
        
        mask_filled = np.zeros_like(gray)
        cv.drawContours(mask_filled, [cnt], -1, 255, -1)
        hull = cv.convexHull(cnt)
        
        # --- 特征提取 (包含新增参数) ---
        feats = {'image_id': fname, 'match_id': base_match_id}
        
        # 基础参数
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        hull_area = cv.contourArea(hull)
        
        # 1. 原始 Morphology (Solidity, Circularity)
        # 这里的 Morph_Circularity 是 4*pi*A / P^2，与 pro.py 中的 logic 一致
        feats['Morph_Circularity'] = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        feats['Morph_Solidity'] = area / hull_area if hull_area > 0 else 0
        
        # 2. 【新增】参数来自 cauliflower-pro.py
        # Compactness (紧凑度): defined as Perimeter / Area in pro.py (max_length / area1)
        feats['Morph_Compactness'] = perimeter / area if area > 0 else 0
        
        # Ellipticity (椭圆率): defined as Short / Long axis
        if len(cnt) >= 5:
            (xc, yc), (d1, d2), angle = cv.fitEllipse(cnt)
            major = max(d1, d2)
            minor = min(d1, d2)
            feats['Morph_Ellipticity'] = minor / major if major > 0 else 0
        else:
            feats['Morph_Ellipticity'] = 0
            
        # Structure - Gaps
        gap_stats, mask_gaps = analyze_gaps_internal(mask_filled, thresh)
        feats.update({'Struct_' + k: v for k, v in gap_stats.items()})
        feats['Struct_FractalDim'] = get_fractal_dimension(gray)
        
        # Color & Texture
        mean_bgr, _ = cv.meanStdDev(img, mask=mask_filled)
        mean_hsv, _ = cv.meanStdDev(cv.cvtColor(img, cv.COLOR_BGR2HSV), mask=mask_filled)
        feats['Color_R'] = mean_bgr[2][0]
        feats['Color_G'] = mean_bgr[1][0]
        feats['Color_B'] = mean_bgr[0][0]
        feats['Color_H'] = mean_hsv[0][0]
        feats['Color_S'] = mean_hsv[1][0]
        feats['Color_V'] = mean_hsv[2][0]
        
        t_c, t_e, t_a = get_glcm_features(gray, mask_filled)
        feats['Text_Contrast'] = t_c
        feats['Text_Entropy'] = t_e
        feats['Text_ASM'] = t_a
        
        # Size
        feats['Size_Area'] = area
        feats['Size_Perimeter'] = perimeter
        
        records.append(feats)
        
        # 可视化输出
        try:
            save_vis_gaps_binary(os.path.join(VIS_DIR, f"{base_name}_gaps.jpg"), mask_gaps)
            save_vis_morphology(os.path.join(VIS_DIR, f"{base_name}_a_morphology.jpg"), img, cnt, hull)
            save_vis_structure_radial(os.path.join(VIS_DIR, f"{base_name}_b_structure.jpg"), img, cnt)
            save_vis_color(os.path.join(VIS_DIR, f"{base_name}_c_color.jpg"), img)
            save_vis_texture_cluster(os.path.join(VIS_DIR, f"{base_name}_d_texture.jpg"), img, mask_filled)
            save_vis_size_style(os.path.join(VIS_DIR, f"{base_name}_e_size.jpg"), img, cnt, hull)
        except Exception as e:
            print(f"Warning: Visual save failed for {fname}: {e}")

    # --- 数据合并与保存 ---
    if records:
        df_img = pd.DataFrame(records)
        
        # 列排序
        cols = sorted(list(df_img.columns), key=lambda x: (x.split('_')[0] if '_' in x else x))
        df_img = df_img[cols]
        
        # 内连接合并，仅保留有效 GR/DD 样本
        final_df = pd.merge(df_img, df_raw_clean, on='match_id', how='inner')
        
        if not final_df.empty:
            final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            print(f"\n[成功] 已保存筛选后的报告: {OUTPUT_CSV}")
            print(f"       包含新参数: Morph_Compactness, Morph_Ellipticity")
            print(f"       仅包含 GR, DD 及有效图片样本数: {len(final_df)}")
        else:
            print("\n[警告] 合并后数据为空！请检查 image_id 和 raw_data ID 是否匹配。")
    else:
        print("未处理任何图片。")