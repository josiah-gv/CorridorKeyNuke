import nuke
import os
import subprocess
import threading
import glob
import sys
import shutil

def run_corridorkey_single_frame():
    node = nuke.thisNode()
    
    def log(msg):
        def update_ui():
            try:
                curr = node.knob('statusText').value()
                node.knob('statusText').setValue(str(msg))
            except:
                pass
        nuke.executeInMainThread(update_ui)
        print(msg)
        
    try:
        # 1. Setup Paths
        work_dir = node.knob('workingDirectory').evaluate()
        if not work_dir:
            log("Error: Working Directory is empty.")
            return
            
        inputs_dir = os.path.join(work_dir, "inputs")
        outputs_dir = os.path.join(work_dir, "outputs")
        plate_dir = os.path.join(inputs_dir, "Plate")
        alpha_dir = os.path.join(inputs_dir, "AlphaHint")
        
        for d in [plate_dir, alpha_dir, outputs_dir]:
            os.makedirs(d, exist_ok=True)
            
        current_frame = nuke.frame()
        
        # 2. Extract Plate and AlphaHint
        log("Exporting Frame " + str(current_frame) + "...")
        
        def build_and_render_writes():
            nuke.root().begin()
            
            write_plate = nuke.nodes.Write(channels="rgb", name="ExportPlate_Temp_" + str(current_frame))
            
            plate_source = node.input(0)
            if plate_source:
                write_plate.setInput(0, plate_source)
            else:
                write_plate.setInput(0, node)
                
            write_plate.knob('file').fromUserText(os.path.join(plate_dir, "input.%05d.exr").replace('\\', '/'))
            try:
                write_plate.knob('datatype').setValue("16 bit half")
                write_plate.knob('compression').setValue("Zip (1 scanline)")
            except:
                pass
                
            write_alpha = nuke.nodes.Write(channels="rgb", name="ExportAlpha_Temp_" + str(current_frame))
            
            alpha_source = None
            if node.input(1) is not None:
                alpha_source = node.input(1)
            else:
                alpha_source = plate_source if plate_source else node
                
            shuffle = nuke.nodes.Shuffle2(name="ExportShuffle_" + str(current_frame))
            shuffle.setInput(0, alpha_source)
            shuffle['in1'].setValue('rgba')
            shuffle['mappings'].setValue([
                (0, 'rgba.alpha', 'rgba.red'),
                (0, 'rgba.alpha', 'rgba.green'),
                (0, 'rgba.alpha', 'rgba.blue'),
                (0, 'rgba.alpha', 'rgba.alpha')
            ])
            
            write_alpha.setInput(0, shuffle)
            write_alpha.knob('file').fromUserText(os.path.join(alpha_dir, "alpha.%05d.exr").replace('\\', '/'))
            try:
                write_alpha.knob('datatype').setValue("16 bit half")
                write_alpha.knob('compression').setValue("Zip (1 scanline)")
            except:
                pass
            
            try:
                nuke.executeMultiple([write_plate, write_alpha], ([current_frame, current_frame, 1],))
                success = True
                error_msg = ""
            except Exception as e:
                success = False
                error_msg = str(e)
            
            nuke.delete(write_plate)
            nuke.delete(write_alpha)
            nuke.delete(shuffle)
            nuke.root().end()
            
            return success, error_msg
            
        success, error_msg = nuke.executeInMainThreadWithResult(build_and_render_writes)
        
        if not success:
            log("Export Error:\\n" + error_msg)
            return
        else:
            log("Export Complete.")
        
        # 3. Execution (Subprocess)
        log("Running CorridorKey ML...")
        
        install_path = node.knob('installPath').value()
        if install_path:
            install_path = os.path.expanduser(install_path)
            
        custom_python = node.knob('customPythonPath').value()
        if custom_python:
            custom_python = os.path.expanduser(custom_python)
        
        python_exe = None
        if custom_python and os.path.exists(custom_python):
            python_exe = custom_python
        elif install_path:
            if sys.platform == "win32":
                guessed = os.path.join(install_path, "venv", "Scripts", "python.exe")
            else:
                guessed = os.path.join(install_path, "venv", "bin", "python")
            
            if os.path.exists(guessed):
                python_exe = guessed

        if not python_exe:
            if shutil.which("python3"):
                python_exe = shutil.which("python3")
            elif shutil.which("python"):
                python_exe = shutil.which("python")
            else:
                python_exe = sys.executable

        if not python_exe:
            log("Error: Could not locate a valid Python executable.")
            return
                
        worker_script = os.path.join(install_path, "headless_nuke_worker.py")
        if not os.path.exists(worker_script):
            log("Error: Could not find headless_nuke_worker.py")
            return
            
        gamma = node.knob('gammaSpace').value()
        despill = node.knob('despillStrength').value()
        auto_desp = node.knob('autoDespeckle').value()
        desp_size = int(node.knob('despeckleSize').value())
        refiner = node.knob('refinerStrength').value()
        
        cmd = [
            python_exe,
            worker_script,
            "--plate_dir", plate_dir,
            "--alpha_dir", alpha_dir,
            "--output_dir", outputs_dir,
            "--start_frame", str(current_frame),
            "--end_frame", str(current_frame),
            "--gamma", gamma.lower(),
            "--despill", str(despill),
            "--despeckle_size", str(desp_size),
            "--refiner_scale", str(refiner)
        ]
        
        if auto_desp:
            cmd.append("--auto_despeckle")
            
        use_shell = sys.platform == "win32" and python_exe == "python"
        result = subprocess.run(cmd, capture_output=True, text=True, shell=use_shell)
        
        if result.returncode != 0:
            log("ML Error:\\n" + str(result.stderr))
            print(result.stdout)
            return
            
        # 4. Post-Processing (Import)
        log("Loading Output...")
        
        def assign_outputs():
            node.begin()
            
            passes = {
                'Processed': os.path.join(outputs_dir, "Processed", "input.%05d.exr").replace('\\', '/'),
                'FG': os.path.join(outputs_dir, "FG", "input.%05d.exr").replace('\\', '/'),
                'Matte': os.path.join(outputs_dir, "Matte", "input.%05d.exr").replace('\\', '/')
            }
            
            read_nodes = {}
            for name, filepath in passes.items():
                read_name = "Read_" + name
                rn = nuke.toNode(read_name)
                if rn is None:
                    rn = nuke.nodes.Read(name=read_name)
                rn['file'].setValue(filepath)
                rn['first'].setValue(int(node['frameStart'].value()))
                rn['last'].setValue(int(node['frameEnd'].value()))
                rn['origfirst'].setValue(int(node['frameStart'].value()))
                rn['origlast'].setValue(int(node['frameEnd'].value()))
                
                read_nodes[name] = rn
                
            out_rgba = nuke.toNode("RGBA")
            out_fg = nuke.toNode("FG_Color")
            out_matte = nuke.toNode("Matte")
            
            # Temporarily disabled per user request to keep the Gizmo as a passthrough
            # out_rgba.setInput(0, read_nodes['Processed'])
            # out_fg.setInput(0, read_nodes['FG'])
            # out_matte.setInput(0, read_nodes['Matte'])
            
            node.end()
            
            node.knob('statusText').setValue("Finished Frame " + str(current_frame))
            
        nuke.executeInMainThread(assign_outputs)
        
    except Exception as e:
        log("Script Error: " + str(e))
        import traceback
        traceback.print_exc()

# Removed threading.Thread(target=run_corridorkey_single_frame).start() since we want to call it directly.
