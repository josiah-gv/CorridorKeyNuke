import nuke
import os
import subprocess
import threading
import glob
import sys
import shutil
import logging
import datetime

# --- File Logger Setup ---
_LOG_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_LOG_DIR, "corridorkey_nuke.log")

def _get_file_logger():
    """Create/return a logger that writes to corridorkey_nuke.log."""
    logger = logging.getLogger("corridorkey_nuke")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(_LOG_FILE, mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def run_corridorkey_single_frame():
    node = nuke.thisNode()
    flog = _get_file_logger()

    # Write a session separator
    flog.info("=" * 60)
    flog.info("Process Single Frame started at %s", datetime.datetime.now().isoformat())
    flog.info("=" * 60)

    def log(msg):
        """Log to Nuke status knob, console, AND log file."""
        flog.info(str(msg))
        def update_ui():
            try:
                node.knob('statusText').setValue(str(msg))
            except:
                pass
        nuke.executeInMainThread(update_ui)
        print(msg)

    try:
        # ----------------------------------------------------------------
        # 1. SETUP PATHS
        # ----------------------------------------------------------------
        work_dir = node.knob('workingDirectory').evaluate()
        if not work_dir:
            log("Error: Working Directory is empty.")
            return

        inputs_dir = os.path.join(work_dir, "inputs")
        outputs_dir = os.path.join(work_dir, "outputs")
        plate_dir = os.path.join(inputs_dir, "Plate")
        alpha_dir = os.path.join(inputs_dir, "AlphaHint")

        for d in [plate_dir, alpha_dir]:
            os.makedirs(d, exist_ok=True)

        # Clean output dirs from previous runs to prevent stale/corrupt files
        if os.path.exists(outputs_dir):
            shutil.rmtree(outputs_dir)
        os.makedirs(outputs_dir, exist_ok=True)

        current_frame = nuke.frame()

        # ----------------------------------------------------------------
        # 2. PRE-PROCESSING — Export Plate and AlphaHint
        # ----------------------------------------------------------------
        log("Exporting Frame " + str(current_frame) + "...")

        def build_and_render_writes():
            nuke.root().begin()

            # --- Write Plate ---
            write_plate = nuke.nodes.Write(
                channels="rgb",
                name="ExportPlate_Temp_" + str(current_frame),
            )

            plate_source = node.input(0)
            if plate_source:
                write_plate.setInput(0, plate_source)
            else:
                write_plate.setInput(0, node)

            write_plate.knob('file').fromUserText(
                os.path.join(plate_dir, "input.%05d.exr").replace('\\', '/')
            )
            try:
                write_plate.knob('datatype').setValue("16 bit half")
                write_plate.knob('compression').setValue("Zip (1 scanline)")
            except:
                pass

            # --- Write AlphaHint ---
            write_alpha = nuke.nodes.Write(
                channels="rgb",
                name="ExportAlpha_Temp_" + str(current_frame),
            )

            alpha_source = None
            if node.input(1) is not None:
                alpha_source = node.input(1)
            else:
                alpha_source = plate_source if plate_source else node

            shuffle = nuke.nodes.Shuffle2(
                name="ExportShuffle_" + str(current_frame),
            )
            shuffle.setInput(0, alpha_source)
            shuffle['in1'].setValue('rgba')
            shuffle['mappings'].setValue([
                (0, 'rgba.alpha', 'rgba.red'),
                (0, 'rgba.alpha', 'rgba.green'),
                (0, 'rgba.alpha', 'rgba.blue'),
                (0, 'rgba.alpha', 'rgba.alpha'),
            ])

            write_alpha.setInput(0, shuffle)
            write_alpha.knob('file').fromUserText(
                os.path.join(alpha_dir, "alpha.%05d.exr").replace('\\', '/')
            )
            try:
                write_alpha.knob('datatype').setValue("16 bit half")
                write_alpha.knob('compression').setValue("Zip (1 scanline)")
            except:
                pass

            # --- Execute ---
            try:
                nuke.executeMultiple(
                    [write_plate, write_alpha],
                    ([current_frame, current_frame, 1],),
                )
                success = True
                error_msg = ""
            except Exception as e:
                success = False
                error_msg = str(e)

            # --- Cleanup temp nodes ---
            nuke.delete(write_plate)
            nuke.delete(write_alpha)
            nuke.delete(shuffle)
            nuke.root().end()

            return success, error_msg

        success, error_msg = nuke.executeInMainThreadWithResult(build_and_render_writes)

        if not success:
            log("Export Error:\n" + error_msg)
            return
        else:
            log("Export Complete.")

        # ----------------------------------------------------------------
        # 3. EXECUTION — Launch ML subprocess
        # ----------------------------------------------------------------
        log("Running CorridorKey ML...")

        install_path = node.knob('installPath').value()
        if install_path:
            install_path = os.path.expanduser(install_path)

        if not install_path or not os.path.isdir(install_path):
            install_path = os.path.expanduser("~/.nuke/CorridorKeyNuke")

        # Worker script lives in GizmoFiles/
        gizmo_files_dir = os.path.join(install_path, "GizmoFiles")
        worker_script = os.path.join(gizmo_files_dir, "headless_nuke_worker.py")
        if not os.path.exists(worker_script):
            log("Error: Could not find headless_nuke_worker.py at:\n" + worker_script)
            return

        # --- Read UI settings ---
        # Gamma: dropdown returns string ("Linear" or "sRGB") or sometimes index
        gamma_raw = node.knob('gammaSpace').value()
        if isinstance(gamma_raw, str):
            gamma = "srgb" if gamma_raw.lower() == "srgb" else "linear"
        else:
            # Index: 0=Linear, 1=sRGB
            gamma = "srgb" if int(gamma_raw) == 1 else "linear"

        despill = node.knob('despillStrength').value()
        auto_desp = node.knob('autoDespeckle').value()
        desp_size = int(node.knob('despeckleSize').value())
        refiner = node.knob('refinerStrength').value()


        # --- Build the subprocess command ---
        # Strategy: try "uv run" first (reliable venv + deps), fall back to
        # direct python executable from the venv.
        uv_available = shutil.which("uv") is not None

        base_args = [
            worker_script,
            "--plate_dir", plate_dir,
            "--alpha_dir", alpha_dir,
            "--output_dir", outputs_dir,
            "--start_frame", str(current_frame),
            "--end_frame", str(current_frame),
            "--gamma", gamma,
            "--despill", str(despill),
            "--despeckle_size", str(desp_size),
            "--refiner_scale", str(refiner),
        ]

        if auto_desp:
            base_args.append("--auto_despeckle")

        if uv_available:
            # uv run uses the project's pyproject.toml for the correct venv
            cmd = ["uv", "run", "--project", install_path, "python"] + base_args
        else:
            # Fallback: find the venv python directly
            custom_python = node.knob('customPythonPath').value()
            if custom_python:
                custom_python = os.path.expanduser(custom_python)

            python_exe = None
            if custom_python and os.path.exists(custom_python):
                python_exe = custom_python
            elif install_path:
                if sys.platform == "win32":
                    guessed = os.path.join(install_path, ".venv", "Scripts", "python.exe")
                else:
                    guessed = os.path.join(install_path, ".venv", "bin", "python")

                if os.path.exists(guessed):
                    python_exe = guessed
                else:
                    # Also check legacy "venv" folder
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

            cmd = [python_exe] + base_args

        # Run the subprocess
        use_shell = sys.platform == "win32"
        cmd_str = " ".join(cmd)
        log("Running CorridorKey ML... (Frame " + str(current_frame) + ", 1/1)")
        flog.info("Command: %s", cmd_str)
        flog.info("CWD: %s", install_path)

        result = subprocess.run(
            cmd, capture_output=True, text=True, shell=use_shell,
            cwd=install_path,
        )

        # Always log subprocess output to file
        if result.stdout:
            flog.info("--- subprocess stdout ---\n%s", result.stdout.rstrip())
            print(result.stdout)
        if result.stderr:
            flog.info("--- subprocess stderr ---\n%s", result.stderr.rstrip())

        if result.returncode != 0:
            stderr_text = result.stderr or "(no stderr)"
            flog.error("Subprocess exited with code %d", result.returncode)
            log("ML Error:\n" + stderr_text)
            print("--- stderr ---")
            print(stderr_text)
            return
        # Clean up inputs to prevent stale frames on next run
        try:
            shutil.rmtree(inputs_dir)
            flog.info("Cleaned up inputs directory: %s", inputs_dir)
        except Exception as cleanup_err:
            flog.warning("Failed to clean inputs: %s", cleanup_err)

        # ----------------------------------------------------------------
        # 4. POST-PROCESSING — Import outputs into Gizmo
        # ----------------------------------------------------------------
        log("Loading Output...")

        def assign_outputs():
            try:
                # Auto-set primary to "Processed RGBA" (index 1) after processing
                node.knob('mainOutputSelect').setValue(1)
                node.knob('statusText').setValue("Finished Frame " + str(current_frame))
            except Exception:
                pass

        nuke.executeInMainThread(assign_outputs)

    except Exception as e:
        log("Script Error: " + str(e))
        import traceback
        tb_str = traceback.format_exc()
        flog.error("Unhandled exception:\n%s", tb_str)
        traceback.print_exc()

