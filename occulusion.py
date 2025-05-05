#!/usr/bin/env python3
import os
import sys
import math
import json
import glob
import numpy as np

# =============================================================================
# VoxelGrid 类及其成员函数（基于之前的 C++ 代码实现）
# =============================================================================

class VoxelGrid:
    class Voxel:
        def __init__(self):
            # 使用字典存储 label 及其计数
            self.labels = {}
            self.count = 0

    def __init__(self):
        self.resolution = 0.0
        self.sizex = 0
        self.sizey = 0
        self.sizez = 0
        self.voxels = []        # 一维列表，按索引存储每个 voxel
        self.occupied = []      # 存储被占用的 voxel 索引
        self.offset = None      # 4D 偏移向量，最后一维固定为 1
        self.occlusions = []    # 每个 voxel 的 occlusion 值
        self.invalid = []       # 无效标记
        self.occludedBy = []    # 存储 ray 穿越时检测到的遮挡 voxel 的索引
        self.occlusionsValid = False

    def initialize(self, resolution, min_pt, max_pt):
        """
        初始化 voxel grid 参数
        :param resolution: 体素分辨率（体素边长）
        :param min_pt: [xmin, ymin, zmin]
        :param max_pt: [xmax, ymax, zmax]
        """
        self.clear()
        self.resolution = resolution
        self.sizex = int(math.ceil((max_pt[0] - min_pt[0]) / resolution))
        self.sizey = int(math.ceil((max_pt[1] - min_pt[1]) / resolution))
        self.sizez = int(math.ceil((max_pt[2] - min_pt[2]) / resolution))
        total_voxels = self.sizex * self.sizey * self.sizez
        self.voxels = [VoxelGrid.Voxel() for _ in range(total_voxels)]
        # 确保 min_pt, max_pt 总在 voxel grid 内
        ox = min_pt[0] - 0.5 * (self.sizex * resolution - (max_pt[0] - min_pt[0]))
        oy = min_pt[1] - 0.5 * (self.sizey * resolution - (max_pt[1] - min_pt[1]))
        oz = min_pt[2] - 0.5 * (self.sizez * resolution - (max_pt[2] - min_pt[2]))
        self.offset = np.array([ox, oy, oz, 1.0], dtype=float)
        self.occlusions = [-1] * total_voxels
        self.occludedBy = [-2] * total_voxels
        self.invalid = [-1] * total_voxels
        print(f"[VoxelGrid.initialize] resolution={resolution}; num. voxels = "
              f"[{self.sizex}, {self.sizey}, {self.sizez}], maxExtent={max_pt}, minExtent={min_pt}")

    def clear(self):
        """将之前占用的 voxel 清零"""
        for idx in self.occupied:
            self.voxels[idx].count = 0
            self.voxels[idx].labels.clear()
        self.occupied = []

    def index(self, i, j, k):
        """将 (i,j,k) 三维索引转换为一维索引"""
        return i + j * self.sizex + k * self.sizex * self.sizey

    def voxel2position(self, i, j, k):
        """返回 voxel 中心点位置（3D 向量）"""
        return np.array([
            self.offset[0] + i * self.resolution + 0.5 * self.resolution,
            self.offset[1] + j * self.resolution + 0.5 * self.resolution,
            self.offset[2] + k * self.resolution + 0.5 * self.resolution
        ], dtype=float)

    def position2voxel(self, pos):
        """将世界坐标 pos 转换为 voxel 的 (i,j,k) 索引"""
        return np.array([
            int((pos[0] - self.offset[0]) / self.resolution),
            int((pos[1] - self.offset[1]) / self.resolution),
            int((pos[2] - self.offset[2]) / self.resolution)
        ], dtype=int)

    def insert(self, p, label):
        """
        向 voxel grid 插入点 p（仅使用 x,y,z 分量）及其 label
        """
        p = np.array(p, dtype=float)
        tp = p - self.offset[:3]
        i = int(math.floor(tp[0] / self.resolution))
        j = int(math.floor(tp[1] / self.resolution))
        k = int(math.floor(tp[2] / self.resolution))
        if i < 0 or j < 0 or k < 0 or i >= self.sizex or j >= self.sizey or k >= self.sizez:
            return
        gidx = self.index(i, j, k)
        if self.voxels[gidx].count == 0:
            self.occupied.append(gidx)
        self.voxels[gidx].labels[label] = self.voxels[gidx].labels.get(label, 0) + 1
        self.voxels[gidx].count += 1
        self.occlusionsValid = False

    def is_occluded(self, i, j, k):
        """判断 voxel (i,j,k) 是否被遮挡"""
        return self.occlusions[self.index(i, j, k)] > -1

    def is_free(self, i, j, k):
        """判断 voxel (i,j,k) 是否为空（未被遮挡）"""
        return self.occlusions[self.index(i, j, k)] == -1

    def is_invalid(self, i, j, k):
        """判断 voxel (i,j,k) 是否无效（需先调用 update_invalid）"""
        idx = self.index(i, j, k)
        if idx >= len(self.invalid):
            return True
        return (self.invalid[idx] > -1) and (self.invalid[idx] != idx)

    def update_occlusions(self):
        """
        对整个 voxel grid 进行遮挡检测，
        更新 occlusions 和 occludedBy 数组。
        """
        total_voxels = self.sizex * self.sizey * self.sizez
        self.occludedBy = [-2] * total_voxels
        self.occlusions = [-1] * total_voxels
        occludedByCalls = 0
        num_shells = min(self.sizex, int(math.ceil(0.5 * self.sizey)))
        for o in range(num_shells):
            # 固定 i = sizex - o - 1
            i = self.sizex - o - 1
            for j in range(self.sizey):
                for k in range(self.sizez):
                    idx = self.index(i, j, k)
                    if self.occludedBy[idx] == -2:
                        occludedByCalls += 1
                        self.occludedBy[idx] = self.occluded_by(i, j, k)
                    self.occlusions[idx] = self.occludedBy[idx]
            # 固定 j = o
            j = o
            for i in range(self.sizex - o - 1):
                for k in range(self.sizez):
                    idx = self.index(i, j, k)
                    if self.occludedBy[idx] == -2:
                        occludedByCalls += 1
                        self.occludedBy[idx] = self.occluded_by(i, j, k)
                    self.occlusions[idx] = self.occludedBy[idx]
            # 固定 j = sizey - o - 1
            j = self.sizey - o - 1
            for i in range(self.sizex - o - 1):
                for k in range(self.sizez):
                    idx = self.index(i, j, k)
                    if self.occludedBy[idx] == -2:
                        occludedByCalls += 1
                        self.occludedBy[idx] = self.occluded_by(i, j, k)
                    self.occlusions[idx] = self.occludedBy[idx]
        self.invalid = self.occlusions.copy()
        self.occlusionsValid = True
        # print(f"occludedBy called {occludedByCalls} times.")

    def update_invalid(self, position):
        """
        使用给定 observation 位置（position，3D 向量）更新 invalid 标记。
        """
        if not self.occlusionsValid:
            self.update_occlusions()
        total_voxels = self.sizex * self.sizey * self.sizez
        self.occludedBy = [-2] * total_voxels
        num_shells = min(self.sizex, int(math.ceil(0.5 * self.sizey)))
        for o in range(num_shells):
            i = self.sizex - o - 1
            for j in range(self.sizey):
                for k in range(self.sizez):
                    idx = self.index(i, j, k)
                    if self.invalid[idx] == -1:
                        continue
                    if self.occludedBy[idx] == -2:
                        self.occludedBy[idx] = self.occluded_by(i, j, k, endpoint=position)
                    self.invalid[idx] = min(self.invalid[idx], self.occludedBy[idx])
            j = o
            for i in range(self.sizex - o - 1):
                for k in range(self.sizez):
                    idx = self.index(i, j, k)
                    if self.invalid[idx] == -1:
                        continue
                    if self.occludedBy[idx] == -2:
                        self.occludedBy[idx] = self.occluded_by(i, j, k, endpoint=position)
                    self.invalid[idx] = min(self.invalid[idx], self.occludedBy[idx])
            j = self.sizey - o - 1
            for i in range(self.sizex - o - 1):
                for k in range(self.sizez):
                    idx = self.index(i, j, k)
                    if self.invalid[idx] == -1:
                        continue
                    if self.occludedBy[idx] == -2:
                        self.occludedBy[idx] = self.occluded_by(i, j, k, endpoint=position)
                    self.invalid[idx] = min(self.invalid[idx], self.occludedBy[idx])

    def occluded_by(self, i, j, k, endpoint=None, visited=None):
        """
        从 voxel (i,j,k) 沿射线追踪，返回第一个遇到的非空 voxel 的索引；
        若无遮挡则返回 -1。
        :param endpoint: 射线终点（3D 向量），默认 [0,0,0]
        :param visited: 可选列表，用于记录遍历过的 voxel 索引
        """
        if endpoint is None:
            endpoint = np.array([0.0, 0.0, 0.0], dtype=float)
        if visited is not None:
            visited.clear()

        # 当前 voxel 的索引
        Pos = [i, j, k]
        startpoint = self.voxel2position(i, j, k)
        halfResolution = 0.5 * self.resolution

        dir_vec = endpoint - startpoint  # 射线方向

        # 初始化 DDA 算法参数
        NextCrossingT = [0.0, 0.0, 0.0]
        DeltaT = [0.0, 0.0, 0.0]
        Step = [0, 0, 0]
        Out = [0, 0, 0]

        for axis in range(3):
            # 防止除 0
            denom = dir_vec[axis] if abs(dir_vec[axis]) > 1e-6 else 1e-6
            if dir_vec[axis] < 0:
                NextCrossingT[axis] = -halfResolution / denom
                DeltaT[axis] = -self.resolution / denom
                Step[axis] = -1
                Out[axis] = 0
            else:
                NextCrossingT[axis] = halfResolution / denom
                DeltaT[axis] = self.resolution / denom
                Step[axis] = 1
                if axis == 0:
                    Out[axis] = self.sizex
                elif axis == 1:
                    Out[axis] = self.sizey
                elif axis == 2:
                    Out[axis] = self.sizez

        endindexes = self.position2voxel(endpoint)
        i_end, j_end, k_end = int(endindexes[0]), int(endindexes[1]), int(endindexes[2])
        cmpToAxis = [2, 1, 2, 1, 2, 2, 0, 0]
        traversed = []

        while True:
            if (Pos[0] < 0 or Pos[0] >= self.sizex or
                Pos[1] < 0 or Pos[1] >= self.sizey or
                Pos[2] < 0 or Pos[2] >= self.sizez):
                break

            idx = self.index(Pos[0], Pos[1], Pos[2])
            if visited is not None:
                visited.append(np.array(Pos, dtype=int))
            if self.voxels[idx].count > 0:
                for t in traversed:
                    self.occludedBy[t] = idx
                return idx

            traversed.append(idx)
            bits = ((NextCrossingT[0] < NextCrossingT[1]) << 2) + \
                   ((NextCrossingT[0] < NextCrossingT[2]) << 1) + \
                   (NextCrossingT[1] < NextCrossingT[2])
            stepAxis = cmpToAxis[bits]
            Pos[stepAxis] += Step[stepAxis]
            NextCrossingT[stepAxis] += DeltaT[stepAxis]
            if Pos[stepAxis] == Out[stepAxis]:
                break
            if Pos[0] == i_end and Pos[1] == j_end and Pos[2] == k_end:
                break

        for t in traversed:
            self.occludedBy[t] = -1
        return -1

# =============================================================================
# 辅助函数
# =============================================================================

def read_velodyne_bin(file_path):
    """
    读取 KITTI 格式的二进制点云文件，
    返回形状为 (N,4) 的 numpy 数组，每行包含 (x,y,z,intensity)
    """
    points = np.fromfile(file_path, dtype=np.float32)
    points = points.reshape(-1, 4)
    return points

def read_label_file(file_path):
    """
    读取 KITTI 格式的标签文件，
    返回 numpy 数组（每个点对应一个 uint32 标签）
    """
    labels = np.fromfile(file_path, dtype=np.uint32)
    return labels

def save_voxel_grid(vg, output_dir, filename, prefix):
    """
    将 voxel grid 信息保存为 .npz 文件
    保存内容包括：
      - resolution, sizex, sizey, sizez, offset
      - counts：每个 voxel 内点数（整型数组，形状 [sizex, sizey, sizez]）
      - occlusions 与 invalid 数组（形状同上）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_file = os.path.join(output_dir, f"{prefix}_{filename}.npz")
    counts = np.zeros((vg.sizex, vg.sizey, vg.sizez), dtype=np.int32)
    for i in range(vg.sizex):
        for j in range(vg.sizey):
            for k in range(vg.sizez):
                idx = vg.index(i, j, k)
                counts[i, j, k] = vg.voxels[idx].count
    occlusions = np.array(vg.occlusions).reshape((vg.sizex, vg.sizey, vg.sizez))
    invalid = np.array(vg.invalid).reshape((vg.sizex, vg.sizey, vg.sizez))
    np.savez(out_file,
             resolution=vg.resolution,
             sizex=vg.sizex, sizey=vg.sizey, sizez=vg.sizez,
             offset=vg.offset,
             counts=counts,
             occlusions=occlusions,
             invalid=invalid)
    print(f"Saved voxel grid to {out_file}")

# =============================================================================
# 主函数：仅处理一针点云，并进行 occluded 与 invalid 检测
# =============================================================================

def main():
    if len(sys.argv) < 4:
        print("Usage: python gen_data.py <config.json> <sequence_dir> <output_voxel_dir>")
        config_file = './waymo_semcity/191862526745161106_1400_000_1420_000/example.cfg'
        sequence_dir = './waymo_semcity/191862526745161106_1400_000_1420_000'
        output_voxel_dir = './waymo_semcity/191862526745161106_1400_000_1420_000/voxels'
    else:
        config_file = sys.argv[1]
        sequence_dir = sys.argv[2]
        output_voxel_dir = sys.argv[3]

    # 读取配置文件（JSON 格式）
    # with open(config_file, 'r') as f:
    #     config = json.load(f)
    voxelSize = 0.4
    minExtent = [-51.2, -51.2, -2]  # 例如 [xmin, ymin, zmin]
    maxExtent = [51.2,51.2, 30]  # 例如 [xmax, ymax, zmax]

    # 在 sequence_dir/velodyne 中查找点云文件（取排序后的第一个）
    velodyne_dir = os.path.join(sequence_dir, "velodyne")
    bin_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
    if len(bin_files) == 0:
        print("No velodyne .bin files found in:", velodyne_dir)
        sys.exit(1)
    scan_file = bin_files[0]
    print("Using scan file:", scan_file)
    points = read_velodyne_bin(scan_file)

    # 尝试加载标签文件（若存在于 sequence_dir/labels 下，与点云文件同名 .label）
    labels_dir = os.path.join(sequence_dir, "labels")
    label_file = os.path.join(labels_dir, os.path.basename(scan_file).replace(".bin", ".label"))
    if os.path.exists(label_file):
        labels = read_label_file(label_file)
        print(f"Loaded {len(labels)} labels from {label_file}")
    else:
        # 若无标签，则全部赋为缺省标签（例如 1）
        labels = np.ones(points.shape[0], dtype=np.uint32)
        print("Label file not found，using default labels (all 1)")

    # 创建 voxel grid，并填充当前扫描的点云（仅使用单针）
    vg = VoxelGrid()
    vg.initialize(voxelSize, minExtent, maxExtent)
    for p, label in zip(points, labels):
        # 仅使用点的 x,y,z 分量
        vg.insert(p[:3], label)

    # 进行遮挡检测与 invalid 更新
    vg.update_occlusions()
    # 这里设置 endpoint 为 [0,0,0]（传感器坐标系下）；可根据实际需求调整
    endpoint = np.array([0.0, 0.0, 0.0], dtype=float)
    vg.update_invalid(endpoint)

    # 保存结果（保存为 .npz 文件，可用 numpy.load 加载）
    # 此处文件名固定为 "scan000000"，如需多个文件可根据需要修改
    output_filename = "scan000000"
    save_voxel_grid(vg, output_voxel_dir, output_filename, "result")

if __name__ == '__main__':
    main()
