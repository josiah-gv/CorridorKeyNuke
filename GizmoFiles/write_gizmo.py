import os

gizmo_path = '/Users/josiahvaughan/.nuke/CorridorKeyNuke/GizmoFiles/CorridorKey.gizmo'

process_current_frame_python = """import nuke
import os
import sys
import threading

def run_single_frame_hook():
    node = nuke.thisNode()
    install_path = node.knob('installPath').value()
    if not install_path:
        install_path = os.path.expanduser("~/.nuke/CorridorKeyNuke")
        
    gizmo_files_dir = os.path.join(install_path, "GizmoFiles")
    if gizmo_files_dir not in sys.path:
        sys.path.insert(0, gizmo_files_dir)
        
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

process_frame_range_python = """import nuke
import os
import sys
import threading

def run_frame_range_hook():
    node = nuke.thisNode()
    install_path = node.knob('installPath').value()
    if not install_path:
        install_path = os.path.expanduser("~/.nuke/CorridorKeyNuke")
        
    gizmo_files_dir = os.path.join(install_path, "GizmoFiles")
    if gizmo_files_dir not in sys.path:
        sys.path.insert(0, gizmo_files_dir)
        
    try:
        import process_frame_range
        import importlib
        importlib.reload(process_frame_range)
        threading.Thread(target=process_frame_range.run_corridorkey_frame_range).start()
    except Exception as e:
        def show_err():
            print("Failed to load script: " + str(e))
            try: node.knob('statusText').setValue("Error: " + str(e))
            except: pass
        nuke.executeInMainThread(show_err)

run_frame_range_hook()
"""

cancel_processing_python = """import nuke
import os
import sys

def run_cancel():
    node = nuke.thisNode()
    node_name = node.fullName()
    
    install_path = node.knob('installPath').value()
    if not install_path:
        install_path = os.path.expanduser("~/.nuke/CorridorKeyNuke")
        
    gizmo_files_dir = os.path.join(install_path, "GizmoFiles")
    if gizmo_files_dir not in sys.path:
        sys.path.insert(0, gizmo_files_dir)
        
    try:
        import process_state
        process_state.cancel_flags[node_name] = True
        if process_state.active_processes.get(node_name):
            try:
                process_state.active_processes[node_name].terminate()
            except Exception:
                pass
            node.knob('statusText').setValue("Cancelling...")
        else:
            node.knob('statusText').setValue("No active process to cancel.")
            
    except Exception as e:
        print("Failed to cancel process: " + str(e))

run_cancel()
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
import threading

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
        gizmo_files_dir = os.path.join(install_dir, "GizmoFiles")
        
        if not os.path.exists(gizmo_files_dir):
            log("Error: Repository not found at " + install_dir + "\\nPlease click 'Update from GitHub' first.")
            return
        
        if gizmo_files_dir not in sys.path:
            sys.path.insert(0, gizmo_files_dir)
        
        import importlib
        try:
            import nuke_installer
            importlib.reload(nuke_installer)
        except ImportError:
            log("Error: nuke_installer.py not found in " + gizmo_files_dir + "\\nPlease click 'Update from GitHub' to download the latest files.")
            return
        
        nuke_installer.run_install(node, install_dir, log)
        
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

# --- Output routing logic ---
# This is called from knobChanged whenever mainOutputSelect or secondaryOutputSelect changes.
# Output type indices: 0=Original, 1=Processed RGBA, 2=FG Only, 3=Matte Only, 4=Preview Comp
# Internal node names: Plate (input), Read_Processed, Read_FG, Read_Matte, Read_Comp

knob_changed_python = """import nuke
node = nuke.thisNode()
k = nuke.thisKnob()

# --- Auto-fill frame range from input ---
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

# --- Output routing ---
# Map dropdown index to internal Read node name
# 0=Original (Plate input), 1=Processed RGBA, 2=FG Only, 3=Matte Only, 4=Preview Comp
OUTPUT_NODE_MAP = {
    0: "Plate",
    1: "Read_Processed",
    2: "Read_FG",
    3: "Read_Matte",
    4: "Read_Comp",
}

def wire_output(output_node_name, type_index):
    try:
        node.begin()
        out_node = nuke.toNode(output_node_name)
        if out_node is None:
            node.end()
            return
        source_name = OUTPUT_NODE_MAP.get(int(type_index), "Plate")
        source_node = nuke.toNode(source_name)
        if source_node:
            out_node.setInput(0, source_node)
        else:
            # Source not created yet (no process run yet), disconnect
            out_node.setInput(0, nuke.toNode("Plate"))
        node.end()
    except:
        try: node.end()
        except: pass

if k.name() == 'mainOutputSelect':
    wire_output("Output1", node.knob('mainOutputSelect').getValue())

if k.name() == 'secondaryOutputSelect':
    wire_output("Output2", node.knob('secondaryOutputSelect').getValue())
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
 addUserKnob {{22 processFrameRangeButton l "Process Frame Range" +STARTLINE T {{
{process_frame_range_python}
 }}}}
 addUserKnob {{22 cancelProcessingButton l "Cancel" -STARTLINE T {{
{cancel_processing_python}
 }}}}
 addUserKnob {{2 workingDirectory l "Working Directory"}}
 workingDirectory "\\[file dirname \\[value root.name]]/CorridorKey/\\[value name]"
 addUserKnob {{3 frameStart l "Frame Range" t "Start Frame"}}
 addUserKnob {{3 frameEnd l "" -STARTLINE t "End Frame"}}
 addUserKnob {{26 statusText l Status T "Ready"}}


 addUserKnob {{20 outputsTab l Outputs}}
 addUserKnob {{4 mainOutputSelect l "Primary Output" M {{Original "Processed RGBA" "FG Only" "Matte Only" "Preview Comp" ""}}}}
 addUserKnob {{26 divider_outputs l "" +STARTLINE T " "}}
 addUserKnob {{4 secondaryOutputSelect l "Secondary Output" M {{Original "Processed RGBA" "FG Only" "Matte Only" "Preview Comp" ""}}}}
 secondaryOutputSelect "Matte Only"

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
  name Output1
  xpos 0
  ypos 200
 }}
 Input {{
  inputs 0
  name AlphaHint
  xpos -200
  ypos -100
  number 1
 }}
 Output {{
  name Output2
  xpos -200
  ypos 200
 }}
 end_group
"""

with open(gizmo_path, 'w') as f:
    f.write(gizmo_template)

print("Done generating Gizmo with correct curly brace strings for callbacks.")
