from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os
import numpy as np


def test_env():
    print("Getting benchmark dictionary...")
    benchmark_dict = benchmark.get_benchmark_dict()
    print("Initializing libero_10 benchmark...")
    benchmark_instance = benchmark_dict["libero_10"]()

    print("Getting task 0...")
    task = benchmark_instance.get_task(0)

    bddl_files_default_path = get_libero_path("bddl_files")
    bddl_file = os.path.join(
        bddl_files_default_path, task.problem_folder, task.bddl_file
    )
    print(f"BDDL file: {bddl_file}")

    if not os.path.exists(bddl_file):
        print(f"Error: BDDL file {bddl_file} does not exist!")
        return

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }

    print("Initializing OffScreenRenderEnv...")
    try:
        env = OffScreenRenderEnv(**env_args)
    except Exception as e:
        print(f"Error initializing environment: {e}")
        import traceback

        traceback.print_exc()
        return

    print("Resetting environment...")
    try:
        env.reset()
    except Exception as e:
        print(f"Error resetting environment: {e}")
        import traceback

        traceback.print_exc()
        return

    print("Stepping environment...")
    try:
        for _ in range(5):
            # Action space for Panda is usually 7 (joints) + 1 (gripper) = 8? Or just 7?
            # The notebook uses [0.] * 7.
            # Let's check env.action_dim if possible, but notebook says 7.
            # Using 7 might be an issue if gripper is included.
            action = [0.0] * 7
            # If action spec expects 8, 7 might fail or warn.
            # Check observation keys
            ob, reward, done, info = env.step(action)
            if _ == 0:
                print("Observation keys:", ob.keys())
                if "agentview_image" not in ob:
                    print("WARNING: 'agentview_image' not found in observations!")
                else:
                    print(f"agentview_image shape: {ob['agentview_image'].shape}")
            print("Step successful")

        # Test set_init_state
        print("Testing set_init_state...")
        init_states = benchmark_instance.get_task_init_states(0)
        print(f"Loaded {len(init_states)} init states.")
        if len(init_states) > 0:
            env.set_init_state(init_states[0])
            print("set_init_state successful")

            # Step again to make sure physics still works
            env.step([0.0] * 7)
            print("Step after set_init_state successful")

    except Exception as e:
        print(f"Error stepping environment: {e}")
        import traceback

        traceback.print_exc()

    env.close()
    print("Done.")


if __name__ == "__main__":
    test_env()
