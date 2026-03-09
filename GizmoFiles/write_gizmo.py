import os

gizmo_path = '/Users/josiahvaughan/.nuke/CorridorKeyNuke/CorridorKey.gizmo'

process_current_frame_python = """import nuke
import os
import sys
import threading

def run_single_frame_hook():
    node = nuke.thisNode()
    install_path = node.knob('installPath').value()
    if not install_path:
        install_path = os.path.expanduser("~/.nuke/CorridorKeyNuke")
        
    if install_path not in sys.path:
        sys.path.append(install_path)
        
    try:
        import process_single_frame
        import importlib
        importlib.reload(process_single_frame)
        threading.Thread(target=process_single_frame.run_corridorkey_single_frame).start()
    except Exception as e:
        def show_err():
            print("Failed to load script: " + str(e))
            try: node.knob('statusText').setValue("Error: " + str(e))
            except: pass
        nuke.executeInMainThread(show_err)

run_single_frame_hook()
"""

update_button_python = """import nuke
import os
import subprocess
import threading
import shutil

def run_update():
    node = nuke.thisNode()
    def log(msg):
        def update_ui():
            try:
                current = node.knob('installLog').value()
                if current is None: current = ""
                node.knob('installLog').setValue(current + str(msg) + "\\n")
            except:
                pass
        nuke.executeInMainThread(update_ui)
        print(msg)
        
    try:
        def clear_log():
            try: node.knob('installLog').setValue("")
            except: pass
        nuke.executeInMainThread(clear_log)
        
        install_dir = os.path.expanduser("~/.nuke/CorridorKeyNuke")
        log("Checking GitHub...\\nTarget: " + install_dir)
        
        if not os.path.exists(install_dir):
            log("Target directory does not exist. Cloning repository...")
            os.makedirs(os.path.dirname(install_dir), exist_ok=True)
            result = subprocess.run(("git", "clone", "https://github.com/josiah-gv/CorridorKeyNuke.git", install_dir), capture_output=True, text=True)
            if result.returncode != 0:
                log("Git clone failed:\\n" + str(result.stderr))
                return
            log("Clone successful.\\n" + str(result.stdout))
        else:
            log("Repository exists. Pulling latest updates...")
            result = subprocess.run(("git", "-C", install_dir, "pull"), capture_output=True, text=True)
            if result.returncode != 0:
                log("Git pull failed:\\n" + str(result.stderr))
                return
            log("Update successful.\\n" + str(result.stdout))
            
        # Auto-copy the Gizmo to the Nuke path
        source_gizmo = os.path.join(install_dir, "CorridorKey.gizmo")
        dest_dir = os.path.expanduser("~/.nuke/gizmos")
        dest_gizmo = os.path.join(dest_dir, "CorridorKey.gizmo")
        
        if os.path.exists(source_gizmo):
            os.makedirs(dest_dir, exist_ok=True)
            try:
                shutil.copy2(source_gizmo, dest_gizmo)
                log("Successfully copied new Gizmo file to:\\n" + dest_gizmo + "\\n(Restart Nuke to see UI changes)")
            except Exception as cp_err:
                log("Failed to update Gizmo file:\\n" + str(cp_err))
        else:
            log("Warning: CorridorKey.gizmo not found in pulled repo.")
            
    except Exception as e:
        log("Update Error: " + str(e))

threading.Thread(target=run_update).start()
"""

setup_env_python = """import nuke
import os
import sys
import subprocess
import threading
import shutil

def run_env_setup():
    node = nuke.thisNode()
    
    def log(msg):
        def update_ui():
            try:
                current = node.knob('installLog').value()
                if current is None: current = ""
                node.knob('installLog').setValue(current + str(msg) + "\\n")
            except: pass
        nuke.executeInMainThread(update_ui)
        print(msg)
        
    try:
        def clear_log():
            try: node.knob('installLog').setValue("")
            except: pass
        nuke.executeInMainThread(clear_log)
        
        install_dir = os.path.expanduser("~/.nuke/CorridorKeyNuke")
        if not os.path.exists(install_dir):
            log("Error: Repository not found at " + install_dir + "\\nPlease run 'Update from GitHub' first.")
            return
            
        venv_dir = os.path.join(install_dir, "venv")
        
        # 1. Find Base Python
        base_python = shutil.which("python3") or shutil.which("python")
        if not base_python:
            log("Error: Could not find Python 3 installed on your system.\\nPlease install Python 3.10+.")
            return
            
        log("Using base python: " + base_python)
        
        # 2. Create Venv
        if not os.path.exists(venv_dir):
            log("Creating virtual environment (this may take a minute)...")
            res = subprocess.run([base_python, "-m", "venv", venv_dir], capture_output=True, text=True)
            if res.returncode != 0:
                log("Venv creation failed:\\n" + res.stderr)
                return
            log("Virtual environment created successfully.")
        else:
            log("Virtual environment already exists.")
            
        # 3. Determine Venv Python path
        if sys.platform == "win32":
            venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            venv_py = os.path.join(venv_dir, "bin", "python")
            
        if not os.path.exists(venv_py):
            log("Error: Virtual environment python executable not found at " + venv_py)
            return
            
        # 4. Install Requirements
        req_file = os.path.join(install_dir, "requirements.txt")
        if not os.path.exists(req_file):
            log("Error: requirements.txt not found in repository.")
            return
            
        log("Upgrading pip...")
        subprocess.run([venv_py, "-m", "pip", "install", "--upgrade", "pip"], capture_output=True)
            
        log("Installing dependencies from requirements.txt... (This will take a while)")
        
        # Run pip install
        # We use Popen to stream output to the log
        process = subprocess.Popen(
            [venv_py, "-m", "pip", "install", "-r", req_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in process.stdout:
            log(line.strip())
            
        process.wait()
        
        if process.returncode != 0:
            log("\\nDependency installation failed with code " + str(process.returncode))
            return
            
        # 5. Download Model Weights
        import urllib.request
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        ckpt_dir = os.path.join(install_dir, "CorridorKeyModule", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_file = os.path.join(ckpt_dir, "CorridorKey.pth")
        
        if not os.path.exists(ckpt_file):
            log("\\nDownloading CorridorKey inference weights (~300MB)...\\nThis may take a minute and Nuke may appear frozen. Please wait.")
            url = "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
            try:
                urllib.request.urlretrieve(url, ckpt_file)
                log("Weights downloaded successfully.")
            except Exception as dl_err:
                log("Failed to download weights: " + str(dl_err))
                return
        else:
            log("\\nInference weights already exist.")
            
        log("\\nEnvironment Setup Complete! ML features are now ready to use.")
        
    except Exception as e:
        log("Setup Error: " + str(e))
        import traceback
        traceback.print_exc()

threading.Thread(target=run_env_setup).start()
"""

on_create_python = """import os
import nuke
install_dir = os.path.expanduser("~/.nuke/CorridorKeyNuke")
if os.path.exists(install_dir):
    try:
        nuke.thisNode().knob('installPath').setValue(install_dir.replace('\\\\', '/'))
    except:
        pass
"""

knob_changed_python = """import nuke
node = nuke.thisNode()
k = nuke.thisKnob()

if k.name() in ('inputChange', 'showPanel'):
    try:
        p = node.input(0)
        if p:
            node.knob('frameStart').setValue(int(p.firstFrame()))
            node.knob('frameEnd').setValue(int(p.lastFrame()))
        else:
            node.knob('frameStart').setValue(int(nuke.root().firstFrame()))
            node.knob('frameEnd').setValue(int(nuke.root().lastFrame()))
    except:
        pass
"""

gizmo_template = f"""Gizmo {{
 inputs 2
 tile_color 0x000000ff
 note_font "Arial Bold"
 note_font_size 11
 note_font_color 0xf9f024ff
 label "       <img src=\\"Corridor_Icon.png\\" width=\\"30\\" height=\\"30\\">"
 help "CorridorKey Integration for Nuke"
 addUserKnob {{20 mainTab l "Main"}}
 addUserKnob {{22 processCurrentFrameButton l "Process Single Frame" T {{
{process_current_frame_python}
 }}}}
 addUserKnob {{22 processFrameRangeButton l "Process Frame Range" T "print('Processing Frame Range...')" +STARTLINE}}
 addUserKnob {{2 workingDirectory l "Working Directory"}}
 workingDirectory "\\[file dirname \\[value root.name]]/CorridorKey/\\[value name]"
 addUserKnob {{3 frameStart l "Frame Range" t "Start Frame"}}
 addUserKnob {{3 frameEnd l "" -STARTLINE t "End Frame"}}
 addUserKnob {{26 statusText l Status T "Ready"}}

 addUserKnob {{20 outputsTab l Outputs}}
 addUserKnob {{6 enableMatteOutput l "Enable Matte Output" +STARTLINE}}
 addUserKnob {{6 enableFGColorOutput l "Enable FG Color Output" +STARTLINE}}

 addUserKnob {{20 settingsTab l "CorridorKey Settings"}}
 addUserKnob {{4 gammaSpace l "Gamma Space" M {{Linear sRGB ""}}}}
 addUserKnob {{26 "" l " " T " "}}
 addUserKnob {{7 despillStrength l "Despill Strength" R 0 10}}
 addUserKnob {{6 autoDespeckle l "Auto-Despeckle" +STARTLINE}}
 addUserKnob {{7 despeckleSize l "Despeckle Size" -STARTLINE R 0 100}}
 addUserKnob {{7 refinerStrength l "Refiner Strength" R 0 5}}
 refinerStrength 1

 addUserKnob {{20 pathsTab l "Paths & Install"}}
  addUserKnob {{22 updateButton l "Update from GitHub" +STARTLINE T {{
{update_button_python}
  }}}}
  addUserKnob {{22 setupEnvButton l "Setup Environment" -STARTLINE T {{
{setup_env_python}
  }}}}
 addUserKnob {{43 installLog l "Install Log"}}
 addUserKnob {{26 "" l " " T " "}}
 addUserKnob {{2 installPath l "CorridorKey Install Path"}}
 addUserKnob {{2 customPythonPath l "Custom Python Path"}}
 addUserKnob {{2 customScriptPath l "Custom Script Path"}}
 onCreate {{
{on_create_python}
 }}
 knobChanged {{
{knob_changed_python}
 }}
}}
 Input {{
  inputs 0
  name Plate
  xpos 0
  ypos -100
 }}
 Output {{
  name RGBA
  xpos -200
  ypos 100
 }}
 Input {{
  inputs 0
  name AlphaHint
  xpos -200
  ypos -100
  number 1
 }}
set Nalpha [stack 0]
 Output {{
  name Matte
  xpos 0
  ypos 100
 }}
push $Nalpha
 Output {{
  name FG_Color
  xpos 150
  ypos 100
 }}
 end_group
"""

with open(gizmo_path, 'w') as f:
    f.write(gizmo_template)

print("Done generating Gizmo with correct curly brace strings for callbacks.")
