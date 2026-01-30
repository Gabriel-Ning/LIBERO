import sys
import os
import argparse


def setup_libero_paths():
    """
    Detects the current location of the libero package and updates
    the default configuration to point to it.
    """
    try:
        try:
            import libero
        except ImportError:
            # Fallback: try to add the local directory to sys.path
            # Assumes the script is in scripts/ and the package is in libero/
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_script_dir)
            libero_pkg_dir = os.path.join(project_root, "libero")
            if os.path.exists(libero_pkg_dir):
                sys.path.insert(0, libero_pkg_dir)
                import libero
            else:
                raise

        from libero import set_libero_default_path

        print(f"[INFO] Found libero package at: {libero.__file__}")

        # Calculate the directory containing the package
        # libero.__file__ is .../libero/libero/__init__.py
        # We want the directory containing __init__.py
        target_path = os.path.dirname(os.path.abspath(libero.__file__))

        print(f"[INFO] Setting LIBERO default path to: {target_path}")
        print("-" * 50)

        # This function handles the config update and warning messages
        set_libero_default_path(target_path)

        print("-" * 50)
        print("[SUCCESS] Libero path configuration updated successfully!")
        print(f"         Your ~/.libero/config.yaml now points to this installation.")

    except ImportError:
        print("[ERROR] Could not import 'libero'.")
        print(
            "        Please ensure you have installed the package via 'pip install -e .'"
        )
        print("        and that you are in the correct virtual environment.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Libero default paths.")
    args = parser.parse_args()

    setup_libero_paths()
