import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2hsv
from skimage.feature import local_binary_pattern
from skimage import img_as_float
import time
import cv2


# 生成原始区域集的函数，用Felzenszwalb图像分割算法，每个区域都有一个编号
def generate_segments(image, scale, sigma, min_size):
    im_mask = felzenszwalb(img_as_float(image), scale=scale, sigma=sigma, min_size=min_size)
    im_orig = np.append(image, np.zeros(image.shape[:2])[:, :, np.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask
    return im_orig


# 计算颜色直方图
def calculate_color_histogram(image):
    BINS = 25
    hist = np.array([])
    for color_channel in (0, 1, 2):
        c = image[:, color_channel]
        hist = np.concatenate([hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])
    hist = hist / len(image)
    return hist


# 计算纹理直方图
def calculate_texture_histogram(image):
    BINS = 10
    hist = np.array([])
    for color_channel in (0, 1, 2):
        fd = image[:, color_channel]
        hist = np.concatenate([hist] + [np.histogram(fd, BINS, (0.0, 1.0))[0]])
    hist = hist / len(image)
    return hist


# 提取区域的尺寸，颜色和纹理特征
def extract_regions(image):
    R = {}
    hsv = rgb2hsv(image[:, :, :3])
    for y, row in enumerate(image):
        for x, (r, g, b, l) in enumerate(row):
            if l not in R:
                R[l] = {"min_x": float('inf'), "min_y": float('inf'), "max_x": 0, "max_y": 0, "labels": [l]}
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y
    tex_grad = calculate_texture_gradient(image)
    for k, v in list(R.items()):
        masked_pixels = hsv[:, :, :][image[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels) // 4
        R[k]["hist_c"] = calculate_color_histogram(masked_pixels)
        R[k]["hist_t"] = calculate_texture_histogram(tex_grad[:, :][image[:, :, 3] == k])
    return R


# 找邻居的函数
def extract_neighbours(regions):
    def intersect(a, b):
        return (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"]) or \
               (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) or \
               (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) or \
               (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"])

    R = list(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))
    return neighbours


# 合并两个区域的函数
def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


# 计算颜色相似度
def sim_color(r1, r2):
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


# 计算纹理相似度
def sim_texture(r1, r2):
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


# 计算尺寸相似度
def sim_size(r1, r2, imsize):
    return 1.0 - (r1["size"] + r2["size"]) / imsize


# 计算填充相似度
def sim_fill(r1, r2, imsize):
    bbsize = (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"])) * \
             (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


# 计算两个区域的相似度
def calc_similarity(r1, r2, imsize):
    return (sim_color(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))


# 计算纹理梯度
def calculate_texture_gradient(image):
    ret = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    for color_channel in (0, 1, 2):
        ret[:, :, color_channel] = local_binary_pattern(image[:, :, color_channel], 8, 1.0)
    return ret


# selective_search主函数
def selective_search(image, scale=1.0, sigma=0.8, min_size=50):
    assert image.shape[2] == 3, "判读输入是不是三通道的图片，如果不是的话不可"
    img = generate_segments(image, scale, sigma, min_size)
    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = extract_regions(img)
    neighbours = extract_neighbours(R)
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_similarity(ar, br, imsize)

    while S != {}:
        i, j = sorted(S.items(), key=lambda x: x[1])[-1][0]
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])
        key_to_delete = [k for k, v in S.items() if i in k or j in k]
        for k in key_to_delete:
            del S[k]
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = calc_similarity(R[t], R[n], imsize)

    regions = []
    for k, r in R.items():
        regions.append({
            'rect': (r['min_x'], r['min_y'], r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions


def filter_regions(regions):
    candidates = set()
    for r in regions:
        # 排除重复的候选区
        if r['rect'] in candidates:
            continue
        # 排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)
        if r['size'] < 2000:
            continue
        # 排除扭曲的候选区域边框  即只保留近似正方形的
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])
    return candidates


def plot(image, rectangles, scale, sigma, min_size):
    fig, ax = plt.subplots(figsize=(8, 8))
    img_draw = image.copy()

    for x, y, w, h in rectangles:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    # 添加参数标签
    text = f"Scale: {scale}, Sigma: {sigma}, Min Size: {min_size}"
    text_x = (img_draw.shape[1] - len(text) * 7) / 2  # 计算使文本居中的 x 坐标
    ax.text(text_x, 10, text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7), va='bottom',
            ha='left')

    ax.imshow(img_draw)
    ax.axis('off')
    plt.show()


def main():
    img = cv2.imread('astronaut.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 选择性搜索参数
    scale = 500
    sigma = 0.9
    min_size = 10

    img_lbl, regions = selective_search(img, scale=scale, sigma=sigma, min_size=min_size)
    candidates = filter_regions(regions)  # 使用提供的筛选函数来获取候选区域
    plot(img, candidates, scale, sigma, min_size)


start = time.time()
main()
end = time.time()
time = end - start
print("处理时间", time, "秒")