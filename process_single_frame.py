import nuke
import os
import subprocess
import threading
import glob

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
        log(f"Exporting Frame {current_frame}...")
        
        # We need to create nodes inside the group
        node.begin()
        
        # --- Plate Export ---
        plate_in = nuke.toNode("Plate")
        write_plate = nuke.nodes.Write(channels="rgb")
        write_plate.setInput(0, plate_in)
        write_plate.knob('file').fromUserText(os.path.join(plate_dir, "input.%05d.exr").replace('\\', '/'))
        try:
            write_plate.knob('datatype').setValue("16 bit half")
            write_plate.knob('compression').setValue("Zip (1 scanline)")
        except:
            pass
        
        # --- AlphaHint Export ---
        alpha_in = nuke.toNode("AlphaHint")
        
        # Determine source of AlphaHint (Input 1 or Fallback to Input 0 Plate)
        alpha_source = None
        if node.input(1) is not None:
            alpha_source = alpha_in
        else:
            alpha_source = plate_in
            
        # Shuffle Alpha to RGB so CorridorKey receives a B&W image
        shuffle = nuke.nodes.Shuffle2()
        shuffle.setInput(0, alpha_source)
        # Set all RGB outputs to the incoming Alpha channel
        shuffle['in1'].setValue('rgba')
        shuffle['mappings'].setValue([
            (0, 'rgba.alpha', 'rgba.red'),
            (0, 'rgba.alpha', 'rgba.green'),
            (0, 'rgba.alpha', 'rgba.blue'),
            (0, 'rgba.alpha', 'rgba.alpha')
        ])
        
        write_alpha = nuke.nodes.Write(channels="rgb")
        write_alpha.setInput(0, shuffle)
        write_alpha.knob('file').fromUserText(os.path.join(alpha_dir, "alpha.%05d.exr").replace('\\', '/'))
        try:
            write_alpha.knob('datatype').setValue("16 bit half")
            write_alpha.knob('compression').setValue("Zip (1 scanline)")
        except:
            pass
        
        # Execute Writes
        nuke.execute([write_plate, write_alpha], current_frame, current_frame)
        
        # Cleanup temp nodes
        nuke.delete(write_plate)
        nuke.delete(write_alpha)
        nuke.delete(shuffle)
        
        node.end()
        
        # 3. Execution (Subprocess)
        log("Running CorridorKey ML...")
        
        # Find the python executable
        install_path = node.knob('installPath').value()
        custom_python = node.knob('customPythonPath').value()
        
        python_exe = "python" # Default fallback
        if custom_python and os.path.exists(custom_python):
            python_exe = custom_python
        elif install_path:
            # Guess venv path
            if sys.platform == "win32":
                guessed = os.path.join(install_path, "venv", "Scripts", "python.exe")
            else:
                guessed = os.path.join(install_path, "venv", "bin", "python")
            if os.path.exists(guessed):
                python_exe = guessed
                
        # Find worker script
        worker_script = os.path.join(install_path, "headless_nuke_worker.py")
        if not os.path.exists(worker_script):
            log(f"Error: Could not find {worker_script}")
            return
            
        # UI Settings
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
            
        log(f"Command: {' '.join(cmd)}")
        
        # Need shell=True on Windows usually if not absolute path
        use_shell = sys.platform == "win32" and python_exe == "python"
        
        result = subprocess.run(cmd, capture_output=True, text=True, shell=use_shell)
        
        if result.returncode != 0:
            log(f"ML Error:\\n{result.stderr}")
            print(result.stdout)
            return
            
        # 4. Post-Processing (Import)
        log("Loading Output...")
        
        def assign_outputs():
            node.begin()
            
            # Find or Create Read nodes for each pass inside Gizmo
            # Pass names: Processed (RGBA), FG, Matte
            
            passes = {
                'Processed': os.path.join(outputs_dir, "Processed", "input.%05d.exr").replace('\\', '/'),
                'FG': os.path.join(outputs_dir, "FG", "input.%05d.exr").replace('\\', '/'),
                'Matte': os.path.join(outputs_dir, "Matte", "input.%05d.exr").replace('\\', '/')
            }
            
            read_nodes = {}
            for name, filepath in passes.items():
                read_name = f"Read_{name}"
                rn = nuke.toNode(read_name)
                if rn is None:
                    rn = nuke.nodes.Read(name=read_name)
                rn['file'].setValue(filepath)
                rn['first'].setValue(int(node['frameStart'].value()))
                rn['last'].setValue(int(node['frameEnd'].value()))
                rn['origfirst'].setValue(int(node['frameStart'].value()))
                rn['origlast'].setValue(int(node['frameEnd'].value()))
                # Frame padding is 5
                
                read_nodes[name] = rn
                
            # Disconnect existing routing
            out_rgba = nuke.toNode("RGBA")
            out_fg = nuke.toNode("FG_Color")
            out_matte = nuke.toNode("Matte")
            
            out_rgba.setInput(0, read_nodes['Processed'])
            out_fg.setInput(0, read_nodes['FG'])
            out_matte.setInput(0, read_nodes['Matte'])
            
            # Turn on visibility for outputs to match Main settings
            # We don't need to do anything specifically for the nodes internally to be "visible", 
            # as long as they are connected to Outputs, the Gizmo knobs dictate their visibility 
            # from the outside.
            
            node.end()
            
            node.knob('statusText').setValue(f"Finished Frame {current_frame}")
            
        nuke.executeInMainThread(assign_outputs)
        
    except Exception as e:
        log("Script Error: " + str(e))
        import traceback
        traceback.print_exc()

threading.Thread(target=run_corridorkey_single_frame).start()
