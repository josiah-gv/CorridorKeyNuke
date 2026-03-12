import os
import sys
import subprocess
import shutil
import urllib.request
import ssl
import nuke

def run_install(node, install_dir, log_func):
    log_func("Starting Environment Setup...")

    # Find a base Python to create the venv with
    python_exe = shutil.which("python3") or shutil.which("python")
    if not python_exe:
        log_func("Error: Could not find Python 3 installed on your system.\nPlease install Python 3.10+.")
        return
    log_func(f"Using base python: {python_exe}")

    venv_dir = os.path.join(install_dir, "venv")
    if not os.path.exists(venv_dir):
        log_func(f"Creating virtual environment using {python_exe}...")
        result = subprocess.run([python_exe, "-m", "venv", venv_dir], capture_output=True, text=True)
        if result.returncode != 0:
            log_func("Venv creation failed:\n" + str(result.stderr))
            return
        log_func("Venv created.")
    else:
        log_func("Venv already exists, skipping creation.")

    log_func("Ensuring 'uv' is installed...")
    if sys.platform == "win32":
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
        uv_exe_path = os.path.join(venv_dir, "Scripts", "uv.exe")
    else:
        pip_exe = os.path.join(venv_dir, "bin", "pip")
        uv_exe_path = os.path.join(venv_dir, "bin", "uv")

    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        uv_exe = ["uv"]
    except OSError:
        log_func("'uv' not found globally. Installing 'uv' via venv pip...")
        try:
            subprocess.run([pip_exe, "install", "uv"], capture_output=True, text=True, check=True)
            uv_exe = [uv_exe_path]
            log_func("'uv' installed successfully into venv.")
        except Exception as e:
            log_func(f"Failed to install 'uv': {e}")
            return

    toml_file = os.path.join(install_dir, "pyproject.toml")
    if os.path.exists(toml_file):
        # We need to run uv sync inside the install directory to use the pyproject.toml
        # We specify the python executable for the environment using VIRTUAL_ENV
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = venv_dir
        
        log_func("Downloading and installing dependencies with uv... (This may take a few minutes)")
        cmd = uv_exe + ["sync"]
        result = subprocess.run(cmd, cwd=install_dir, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            log_func("UV sync failed:\n" + str(result.stderr))
        else:
            log_func("Dependencies installed via uv.")
    else:
        log_func("pyproject.toml not found! Skipping dependencies.")

    checkpoints_dir = os.path.join(install_dir, "CorridorKeyModule", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_path = os.path.join(checkpoints_dir, "CorridorKey.pth")

    if not os.path.exists(model_path):
        log_func("Downloading CorridorKey_v1.0.pth model to checkpoints...")
        model_url = "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
        try:
            # Bypass macOS SSL certificate errors in embedded Python
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(model_url, context=ctx) as response, open(model_path, 'wb') as out_file:
                out_file.write(response.read())
            log_func("Download complete.")
        except Exception as e:
            log_func("Error downloading model: " + str(e))
            return
    else:
        log_func("Model already exists.")

    def finish():
        log_func("\nEnvironment Setup Complete! ML features are now ready to use.")
        if node:
            node['installPath'].setValue(install_dir.replace('\\', '/'))

    nuke.executeInMainThread(finish)
