import os
import torch
import torchvision
from PIL import Image
import imageio
import h5py
import numpy as np
from termcolor import colored
from libero.libero import benchmark, get_libero_path, set_libero_default_path
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.utils.download_utils as download_utils
from libero.libero.utils.dataset_utils import get_dataset_info


def run_notebook():
    print("=== Cell 1: Imports ===")
    # Done above

    print("\n=== Cell 2: Default file paths ===")
    benchmark_root_path = get_libero_path("benchmark_root")
    init_states_default_path = get_libero_path("init_states")
    datasets_default_path = get_libero_path("datasets")
    bddl_files_default_path = get_libero_path("bddl_files")
    print("Default benchmark root path: ", benchmark_root_path)
    print("Default dataset root path: ", datasets_default_path)
    print("Default bddl files root path: ", bddl_files_default_path)

    print("\n=== Cell 3: Set custom path (and revert) ===")
    set_libero_default_path(os.path.join(os.path.expanduser("~"), "custom_project"))
    print("Custom path set.")
    # Revert
    set_libero_default_path()
    print("Reverted to default path.")

    print("\n=== Cell 4: Available benchmarks ===")
    benchmark_dict = benchmark.get_benchmark_dict()
    print(benchmark_dict)
    for _ in range(len(benchmark_dict)):
        # dict keys are not integers, this loop in notebook is kinda weird if it expects integer indexing on dict
        # but let's see if benchmark_dict is list or dict. It is dict.
        # Notebook code: for _ in range(len(benchmark_dict)): print(benchmark_dict[_])
        # This will fail with KeyError 0 if benchmark_dict keys are strings.
        # I will fix this logic here to iterate over values.
        pass
    for k, v in benchmark_dict.items():
        print(f"{k}: {v}")

    print("\n=== Cell 5: Check integrity of benchmarks ===")
    benchmark_instance = benchmark_dict["libero_10"]()
    num_tasks = benchmark_instance.get_num_tasks()
    print(f"{num_tasks} tasks in the benchmark {benchmark_instance.name}: ")
    task_names = benchmark_instance.get_task_names()
    print("The benchmark contains the following tasks:")
    for i in range(num_tasks):
        task_name = task_names[i]
        task = benchmark_instance.get_task(i)
        bddl_files_default_path = get_libero_path(
            "bddl_files"
        )  # Refresh path just in case
        bddl_file = os.path.join(
            bddl_files_default_path, task.problem_folder, task.bddl_file
        )
        print(f"\t {task_name}, detail definition stored in {bddl_file}")
        if not os.path.exists(bddl_file):
            print(
                colored(
                    f"[error] bddl file {bddl_file} cannot be found. Check your paths",
                    "red",
                )
            )
        else:
            print("OK")

    print("\n=== Cell 6: Check integrity of init files ===")
    init_states_default_path = get_libero_path("init_states")
    print("The benchmark contains the following tasks:")
    for i in range(num_tasks):
        task_name = task_names[i]
        task = benchmark_instance.get_task(i)
        init_states_path = os.path.join(
            init_states_default_path, task.problem_folder, task.init_states_file
        )
        if not os.path.exists(init_states_path):
            print(
                colored(
                    f"[error] the init states {init_states_path} cannot be found. Check your paths",
                    "red",
                )
            )
        else:
            print(f"Init state file found: {init_states_path}")

    print(f"An example of init file is named like this: {task.init_states_file}")

    # Load torch init files
    try:
        init_states = benchmark_instance.get_task_init_states(0)
        print(init_states.shape)
    except Exception as e:
        print(f"Failed to load init states: {e}")

    print("\n=== Cell 7: Visualize init states ===")
    task_id = 9
    task = benchmark_instance.get_task(task_id)
    env_args = {
        "bddl_file_name": os.path.join(
            bddl_files_default_path, task.problem_folder, task.bddl_file
        ),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    try:
        env = OffScreenRenderEnv(**env_args)
        init_states = benchmark_instance.get_task_init_states(task_id)
        env.seed(0)

        images = []
        env.reset()
        # Limit to 2 states to save time
        for eval_index in range(min(2, len(init_states))):
            env.set_init_state(init_states[eval_index])
            for _ in range(5):
                obs, _, _, _ = env.step([0.0] * 7)
            # Check key
            if "agentview_image" in obs:
                img = obs["agentview_image"]
            elif "agentview_rgb" in obs:
                img = obs["agentview_rgb"]
            else:
                print("Warning: No agentview image found")
                img = np.zeros((128, 128, 3))

            images.append(torch.from_numpy(img).permute(2, 0, 1))

        print("Generated images.")
        env.close()
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== Cell 8: Download datasets ===")
    download_dir = get_libero_path("datasets")
    # FIX: Use libero_10 instead of libero_spatial to match validity check
    datasets = "libero_10"

    libero_datasets_exist = download_utils.check_libero_dataset(
        download_dir=download_dir
    )

    print(f"Downloading {datasets} to {download_dir}...")

    # User specific file check
    specific_demo_file = os.path.join(
        download_dir,
        "libero_10",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5",
    )

    if os.path.exists(specific_demo_file):
        print(f"Specific demo file found: {specific_demo_file}")
        print("Skipping download as requested by user.")
    elif not libero_datasets_exist:
        try:
            download_utils.libero_dataset_download(
                download_dir=download_dir, datasets=datasets, use_huggingface=True
            )
        except Exception as e:
            print(f"Download failed: {e}")
    else:
        print(f"Dataset {datasets} already exists, skipping download.")

    # Check validity for libero_10 tasks
    demo_files = [
        os.path.join(
            datasets_default_path, benchmark_instance.get_task_demonstration(i)
        )
        for i in range(num_tasks)
    ]
    for demo_file in demo_files:
        if not os.path.exists(demo_file):
            # Check if it exists in libero_spatial
            spatial_path = demo_file.replace("libero_10", "libero_spatial")
            if os.path.exists(spatial_path):
                print(f"File found in libero_spatial but not libero_10: {spatial_path}")
            else:
                print(
                    colored(
                        f"[error] demo file {demo_file} cannot be found. Check your paths",
                        "red",
                    )
                )
        else:
            print(f"Demo file found: {demo_file}")

    print("\n=== Cell 9: Demo info and replay ===")
    example_demo_file = None
    if os.path.exists(specific_demo_file):
        print(f"Using user specified demo file: {specific_demo_file}")
        example_demo_file = specific_demo_file
    elif len(demo_files) > 0 and os.path.exists(demo_files[9]):
        example_demo_file = demo_files[9]

    if example_demo_file:
        get_dataset_info(example_demo_file)

        try:
            with h5py.File(example_demo_file, "r") as f:
                # Check available keys in dataset
                print("Dataset keys:", f.keys())
                if "data" in f:
                    print("Data keys:", f["data"].keys())
                    if "demo_0" in f["data"]:
                        print("Demo 0 keys:", f["data/demo_0"].keys())
                        if "obs" in f["data/demo_0"]:
                            print("Obs keys:", f["data/demo_0/obs"].keys())

                # The notebook uses agentview_rgb
                if "data/demo_0/obs/agentview_rgb" in f:
                    images = f["data/demo_0/obs/agentview_rgb"][()]
                    print(f"Loaded {len(images)} images from dataset.")

                    video_writer = imageio.get_writer("output.mp4", fps=60)
                    for image in images:
                        video_writer.append_data(image[::-1])
                    video_writer.close()
                    print("Video saved to output.mp4")
                else:
                    print("agentview_rgb not found in dataset")
        except Exception as e:
            print(f"Demo replay failed: {e}")
    else:
        print("Demo file not found, skipping replay.")


if __name__ == "__main__":
    run_notebook()
