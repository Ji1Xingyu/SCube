import os
import glob

def rename_to_kitti_format(folder_path, extension):
    """
    Renames all files with the given extension in `folder_path` to
    zero-padded KITTI format: 000000.ext, 000001.ext, ...
    Returns the sorted list of new filenames after renaming.
    """
    # Gather all files with the specified extension
    files = glob.glob(os.path.join(folder_path, '*'))
    files.sort()  # Ensure a consistent ordering

    # Rename files in zero-padded ascending order
    new_filenames = []
    for idx, old_path in enumerate(files):
        new_name = f"{idx:06d}{extension}"  # e.g. 000000.bin
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        new_filenames.append(new_name)

    return new_filenames


def generate_identity_poses_txt(output_file, num_poses):
    """
    Generates a poses.txt file with `num_poses` lines of the KITTI-style
    identity transformation (3x4 matrix flattened into one line).
    
    Example line for identity:
    1 0 0 0  0 1 0 0  0 0 1 0
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 每行的单位矩阵（3×4）的扁平化写法（总共12个数字）
    identity_line = "1 0 0 0 0 1 0 0 0 0 1 0\n"
    
    with open(output_file, 'w') as f:
        for _ in range(num_poses):
            f.write(identity_line)


def main():
    # Adjust these paths to match your own dataset structure
    base_dir = "./waymo_semcity/191862526745161106_1400_000_1420_000"
    velodyne_dir = os.path.join(base_dir, "velodyne")
    labels_dir = os.path.join(base_dir, "labels")
    calib_file = os.path.join(base_dir, "calib.txt")  # Not used directly here, but available
    poses_file = os.path.join(base_dir, "poses.txt")

    # 1) Rename Velodyne files to KITTI format
    #    (assuming they are all .bin files)
    velodyne_filenames = rename_to_kitti_format(velodyne_dir, ".bin")
    labels_filenames = rename_to_kitti_format(labels_dir, '.label')
    print(f"Renamed Velodyne files: {velodyne_filenames}")
    print(f"Renamed Label files: {labels_filenames}")

    # 2) (Optional) Rename label files to KITTI format
    #    If your label files are, for example, '.label' extension.
    #    Uncomment or change extension as needed.
    # label_filenames = rename_to_kitti_format(labels_dir, ".label")
    # print(f"Renamed label files: {label_filenames}")

    # 3) Generate poses.txt with identity transformation for each point cloud
    num_pointclouds = len(velodyne_filenames)
    generate_identity_poses_txt(poses_file, num_pointclouds)
    print(f"Generated {poses_file} with {num_pointclouds} identity poses.")


if __name__ == "__main__":
    main()
