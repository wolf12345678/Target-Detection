import os
import sys
import platform
import subprocess
import shutil
import time

# --- Configuration ---
APP_NAME = "区域检测器"
MAIN_SCRIPT = "Main2.py"
ENCRYPTED_MODEL_FILE = "encrypted_model.bin" # The crucial encrypted model file
ICON_WINDOWS = "icon.ico" # Optional: Path to your Windows icon file
ICON_MACOS = "icon.icns"   # Optional: Path to your macOS icon file
OUTPUT_DIR = "dist"
# Add more data files if needed (e.g., images, config files)
# Format: {"source_path": "destination_in_package", ...}
# Destination '.' means the root directory alongside the executable.
EXTRA_DATA_FILES = {
    ENCRYPTED_MODEL_FILE: "."
    # Add other files here if necessary, e.g.:
    # "images/logo.png": "images",
    # "config.json": "."
}

# Add modules that PyInstaller might miss, especially dynamically loaded ones
# These are common for the libraries you are using
HIDDEN_IMPORTS = [
    "torch",
    "torchvision",
    "cryptography",
    "cryptography.hazmat.primitives.kdf.pbkdf2",
    "cryptography.hazmat.backends.openssl", # Or other backend if used
    "cv2",
    "pandas",
    "pandas._libs.tslibs.base",
    "matplotlib.backends.backend_qt5agg",
    "PyQt5.sip",
    "sklearn.utils._typedefs", # Sometimes needed if scikit-learn is implicitly used
    "sklearn.neighbors._typedefs",
    "sklearn.neighbors._quad_tree",
    "sklearn.tree",
    "sklearn.tree._utils"
    # Add more based on specific errors during testing
]
# --- End Configuration ---


def check_pyinstaller():
    """Checks for PyInstaller and installs it if missing."""
    print("--- Checking for PyInstaller ---")
    try:
        import PyInstaller
        print(f"PyInstaller Version: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("PyInstaller not found. Attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("PyInstaller installed successfully!")
            # Need to re-check import after install? Maybe not necessary for subprocess call.
            return True
        except Exception as e:
            print(f"Error installing PyInstaller: {e}", file=sys.stderr)
            print("Please install PyInstaller manually: pip install pyinstaller", file=sys.stderr)
            return False

def check_files():
    """Checks if required files exist before starting."""
    print("--- Checking Required Files ---")
    all_found = True
    if not os.path.exists(MAIN_SCRIPT):
        print(f"Error: Main script '{MAIN_SCRIPT}' not found!", file=sys.stderr)
        all_found = False

    for src, dest in EXTRA_DATA_FILES.items():
        if not os.path.exists(src):
            print(f"Error: Data file '{src}' not found!", file=sys.stderr)
            all_found = False

    # Check optional icon files only if specified
    if platform.system() == "Windows" and ICON_WINDOWS and not os.path.exists(ICON_WINDOWS):
         print(f"Warning: Specified Windows icon '{ICON_WINDOWS}' not found. Skipping icon.")
         # Don't set all_found to False for optional files
    if platform.system() == "Darwin" and ICON_MACOS and not os.path.exists(ICON_MACOS):
         print(f"Warning: Specified macOS icon '{ICON_MACOS}' not found. Skipping icon.")
         # Don't set all_found to False for optional files

    if not all_found:
        print("Please ensure all required files are present before packaging.", file=sys.stderr)
    return all_found


def build_command():
    """Builds the PyInstaller command list."""
    print("--- Configuring Build Command ---")
    system = platform.system()
    pyinstaller_cmd = [
        sys.executable, # Use the current python interpreter to run PyInstaller module
        "-m",
        "PyInstaller",
        "--noconfirm",          # Overwrite output directory without asking
        "--clean",              # Clean PyInstaller cache and remove temporary files
        "--name", APP_NAME,
        "--onedir",             # Create a folder containing the executable (easier debugging)
        # Use --onefile later if needed, but start with --onedir
    ]

    # Platform specific settings
    icon_path = None
    if system == "Windows":
        print("Platform: Windows")
        pyinstaller_cmd.append("--windowed") # No console window
        if ICON_WINDOWS and os.path.exists(ICON_WINDOWS):
            icon_path = ICON_WINDOWS
    elif system == "Darwin": # macOS
        print("Platform: macOS")
        pyinstaller_cmd.append("--windowed") # Creates a .app bundle
        if ICON_MACOS and os.path.exists(ICON_MACOS):
            icon_path = ICON_MACOS
    else:
        print(f"Platform: {system} (using defaults)")
        # No '--windowed' for Linux by default, assumes console needed unless specified

    if icon_path:
        pyinstaller_cmd.extend(["--icon", icon_path])
        print(f"Using icon: {icon_path}")

    # Add Data Files
    print("Adding data files:")
    for src, dest in EXTRA_DATA_FILES.items():
        separator = ";" if system == "Windows" else ":"
        arg = f"{src}{separator}{dest}"
        pyinstaller_cmd.extend(["--add-data", arg])
        print(f"  - {arg}")

    # Add Hidden Imports
    print("Adding hidden imports:")
    for module in HIDDEN_IMPORTS:
        pyinstaller_cmd.extend(["--hidden-import", module])
        print(f"  - {module}")

    # Add the main script
    pyinstaller_cmd.append(MAIN_SCRIPT)

    return pyinstaller_cmd


def run_build(command):
    """Runs the PyInstaller build command."""
    print("\n--- Starting Build Process ---")
    print(f"Executing command:\n{' '.join(command)}")
    start_time = time.time()
    try:
        # Run PyInstaller using subprocess
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        # Print stdout and stderr line by line
        while True:
            output_stdout = process.stdout.readline()
            output_stderr = process.stderr.readline()
            if output_stdout == '' and output_stderr == '' and process.poll() is not None:
                break
            if output_stdout:
                print(output_stdout.strip())
            if output_stderr:
                print(output_stderr.strip(), file=sys.stderr) # Print errors to stderr

        returncode = process.poll()

        end_time = time.time()
        print(f"--- Build Finished (Duration: {end_time - start_time:.2f} seconds) ---")

        if returncode == 0:
            print("\nBuild successful!")
            system = platform.system()
            if system == "Darwin":
                output_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"{APP_NAME}.app"))
            else:
                output_path = os.path.abspath(os.path.join(OUTPUT_DIR, APP_NAME)) # For Windows/Linux .exe is inside this folder
            print(f"Output created in: {output_path}")
            return True
        else:
            print(f"\nError: PyInstaller failed with exit code {returncode}", file=sys.stderr)
            return False

    except Exception as e:
        print(f"\nAn unexpected error occurred during the build process: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the packaging process."""
    print(f"--- Starting {APP_NAME} Packaging ---")

    if not check_pyinstaller():
        sys.exit(1)

    if not check_files():
        sys.exit(1)

    build_cmd = build_command()

    if not run_build(build_cmd):
        print("\nPackaging failed. Please check the error messages above.", file=sys.stderr)
        print("Tips for troubleshooting:")
        print(" - Ensure all dependencies (torch, PyQt5, etc.) are installed correctly in the environment.")
        print(" - Check the PyInstaller output for specific module-not-found errors and add them to HIDDEN_IMPORTS.")
        print(" - Try running the application from the command line after building to see runtime errors.")
        print(" - For complex issues, consider generating a .spec file first ('pyinstaller your_script.py') and editing it.")
        sys.exit(1)

    print("\n--- Packaging Complete ---")

if __name__ == "__main__":
    main()
