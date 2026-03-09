import os
import sys
import subprocess
import urllib.request
import ssl
import nuke

def run_install(node, install_dir, log_func):
    log_func("Starting Phase 3: Heavy Lifting Setup...")

    python_exe = sys.executable
    if "Nuke" in python_exe or "nuke" in python_exe.lower():
        if sys.platform == "win32":
            python_exe = os.path.join(os.path.dirname(sys.executable), "python.exe")
        elif sys.platform == "darwin":
            python_exe = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Frameworks", "Python.framework", "Versions", "Current", "bin", "python3")
            if not os.path.exists(python_exe):
                python_exe = "python3"
        else:
            python_exe = os.path.join(os.path.dirname(sys.executable), "python3")
    if not os.path.exists(python_exe):
        python_exe = "python3"

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

    log_func("Installing dependencies...")
    if sys.platform == "win32":
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        pip_exe = os.path.join(venv_dir, "bin", "pip")

    req_file = os.path.join(install_dir, "requirements.txt")
    if os.path.exists(req_file):
        result = subprocess.run([pip_exe, "install", "-r", req_file], capture_output=True, text=True)
        if result.returncode != 0:
            log_func("Pip install failed:\n" + str(result.stderr))
        else:
            log_func("Dependencies installed.")
    else:
        log_func("requirements.txt not found! Skipping dependencies.")

    checkpoints_dir = os.path.join(install_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_path = os.path.join(checkpoints_dir, "CorridorKey_v1.0.pth")

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
        log_func("CorridorKey Installation Complete!")
        if node:
            node['installPath'].setValue(install_dir.replace('\\\\', '/'))

    nuke.executeInMainThread(finish)
