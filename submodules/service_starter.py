import os
import glob

OPENPOSE_OUTPUT_DIR_NAME = "keypoints"
DEEPSORT_OUTPUT_DIR_NAME = "ID"

os.chdir(os.path.join(__file__, ".."))

input_paths = glob.glob(os.path.join(__file__, "..", "inputs", "*"))
if not input_paths:
    raise (FileNotFoundError("動画が見つかりません. inputsフォルダに動画を配置してください. "))

container_input_paths = [
    f"/app/inputs/{os.path.basename(path)}" for path in input_paths
]

for container_input_path in container_input_paths:
    video_name = os.path.splitext(os.path.basename(container_input_path))[0]
    container_out_dir = f"/app/outputs/{video_name}"

    # OPENPOSE
    os.makedirs(
        os.path.join(__file__, "..", "outputs", video_name, OPENPOSE_OUTPUT_DIR_NAME),
        exist_ok=True,
    )
    os.system(
        f'docker compose run --rm openpose /bin/bash -c "./build/examples/openpose/openpose.bin --video {container_input_path} --render_pose 0 --display 0 --write_json {container_out_dir}/{OPENPOSE_OUTPUT_DIR_NAME}"'
    )

    # DEEPSORT
    os.makedirs(
        os.path.join(__file__, "..", "outputs", video_name, DEEPSORT_OUTPUT_DIR_NAME),
        exist_ok=True,
    )
    os.system(
        f'docker compose run --rm deepsort /bin/bash -c "python object_tracker.py --video {container_input_path} --info {container_out_dir}/{DEEPSORT_OUTPUT_DIR_NAME} --dont_show --count"'
    )
