import nuke
import os
import subprocess
import threading
import glob
import sys
import shutil
import logging
import datetime
import time
import re
import process_state

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


def run_corridorkey_frame_range():
    node = nuke.thisNode()
    node_name = node.fullName()
    process_state.cancel_flags[node_name] = False
    flog = _get_file_logger()

    # Write a session separator
    flog.info("=" * 60)
    flog.info("Process Frame Range started at %s", datetime.datetime.now().isoformat())
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
        # 1. SETUP PATHS & FRAME RANGE
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

        # Read frame range from the node knobs
        frame_start = int(node.knob('frameStart').value())
        frame_end = int(node.knob('frameEnd').value())

        if frame_start > frame_end:
            log("Error: Frame Start (%d) is greater than Frame End (%d)." % (frame_start, frame_end))
            return

        total_frames = frame_end - frame_start + 1

        # ----------------------------------------------------------------
        # 2. PRE-PROCESSING — Export Plate and AlphaHint for full range
        # ----------------------------------------------------------------
        log("Exporting Frames %d-%d (%d frames)..." % (frame_start, frame_end, total_frames))

        def build_and_render_writes():
            nuke.root().begin()

            # --- Write Plate ---
            write_plate = nuke.nodes.Write(
                channels="rgb",
                name="ExportPlate_Temp_Range",
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
                name="ExportAlpha_Temp_Range",
            )

            alpha_source = None
            if node.input(1) is not None:
                alpha_source = node.input(1)
            else:
                alpha_source = plate_source if plate_source else node

            shuffle = nuke.nodes.Shuffle2(
                name="ExportShuffle_Range",
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

            # --- Execute full frame range ---
            try:
                nuke.executeMultiple(
                    [write_plate, write_alpha],
                    ([frame_start, frame_end, 1],),
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
            log("Export Complete. %d frames exported." % total_frames)

        # ----------------------------------------------------------------
        # 3. EXECUTION — Launch ML subprocess with streaming progress
        # ----------------------------------------------------------------
        log("Running CorridorKey ML... (Frames %d-%d)" % (frame_start, frame_end))

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
        gamma_raw = node.knob('gammaSpace').value()
        if isinstance(gamma_raw, str):
            gamma = "srgb" if gamma_raw.lower() == "srgb" else "linear"
        else:
            gamma = "srgb" if int(gamma_raw) == 1 else "linear"

        despill = node.knob('despillStrength').value()
        auto_desp = node.knob('autoDespeckle').value()
        desp_size = int(node.knob('despeckleSize').value())
        refiner = node.knob('refinerStrength').value()

        # --- Build the subprocess command ---
        uv_available = shutil.which("uv") is not None

        base_args = [
            worker_script,
            "--plate_dir", plate_dir,
            "--alpha_dir", alpha_dir,
            "--output_dir", outputs_dir,
            "--start_frame", str(frame_start),
            "--end_frame", str(frame_end),
            "--gamma", gamma,
            "--despill", str(despill),
            "--despeckle_size", str(desp_size),
            "--refiner_scale", str(refiner),
        ]

        if auto_desp:
            base_args.append("--auto_despeckle")

        if uv_available:
            cmd = ["uv", "run", "--project", install_path, "python"] + base_args
        else:
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

        # --- Run subprocess with streaming stdout ---
        use_shell = sys.platform == "win32"
        cmd_str = " ".join(cmd)
        flog.info("Command: %s", cmd_str)
        flog.info("CWD: %s", install_path)

        # Terminate any existing process for this node first
        if process_state.active_processes.get(node_name):
            flog.warning("Terminating previous active process for this node.")
            try:
                process_state.active_processes[node_name].terminate()
            except Exception:
                pass

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, shell=use_shell, cwd=install_path, bufsize=1,
        )
        process_state.active_processes[node_name] = proc

        # Parse stdout line-by-line for per-frame progress
        frames_processed = 0
        frame_start_time = time.time()
        last_sec_per_frame = None

        for line in proc.stdout:
            if process_state.cancel_flags.get(node_name, False):
                flog.info("Cancellation requested via UI.")
                proc.terminate()
                break

            line = line.rstrip()
            if not line:
                continue

            flog.info("worker: %s", line)
            print(line)

            # Detect "Processing: input.XXXXX" lines from the worker
            if "Processing:" in line:
                now = time.time()
                if frames_processed > 0:
                    # Compute sec/frame from the previous frame
                    last_sec_per_frame = now - frame_start_time

                frames_processed += 1
                frame_start_time = now

                # Build status message
                status = "ML Processing... Frame %d/%d" % (frames_processed, total_frames)
                if last_sec_per_frame is not None:
                    status += " (%.1fs/frame)" % last_sec_per_frame
                log(status)

        # Wait for process to finish and capture stderr
        _, stderr_output = proc.communicate()

        process_state.active_processes.pop(node_name, None)

        if process_state.cancel_flags.get(node_name, False):
            log("ML Processing Cancelled by User.")
            process_state.cancel_flags[node_name] = False
            return

        if stderr_output:
            flog.info("--- subprocess stderr ---\n%s", stderr_output.rstrip())

        if proc.returncode != 0:
            stderr_text = stderr_output or "(no stderr)"
            flog.error("Subprocess exited with code %d", proc.returncode)
            log("ML Error:\n" + stderr_text)
            print("--- stderr ---")
            print(stderr_text)
            return

        # Report final timing
        if frames_processed > 0 and last_sec_per_frame is not None:
            log("ML Complete. %d frames processed (%.1fs/frame avg)." % (
                frames_processed,
                last_sec_per_frame,
            ))
        else:
            log("ML Complete. %d frames processed." % frames_processed)

        # Clean up inputs to prevent stale frames on next run
        try:
            shutil.rmtree(inputs_dir)
            flog.info("Cleaned up inputs directory: %s", inputs_dir)
        except Exception as cleanup_err:
            flog.warning("Failed to clean inputs: %s", cleanup_err)

        # ----------------------------------------------------------------
        # 4. POST-PROCESSING — Import outputs into Gizmo
        # ----------------------------------------------------------------
        log("Loading Outputs...")

        def assign_outputs():
            try:
                # Auto-set primary to "Processed RGBA" (index 1) after processing
                node.knob('mainOutputSelect').setValue(1)
                node.knob('statusText').setValue(
                    "Finished Frames %d-%d (%d frames)" % (frame_start, frame_end, total_frames)
                )
            except Exception:
                pass

        nuke.executeInMainThread(assign_outputs)

    except Exception as e:
        log("Script Error: " + str(e))
        import traceback
        tb_str = traceback.format_exc()
        flog.error("Unhandled exception:\n%s", tb_str)
        traceback.print_exc()


def _build_sequence_path(file_path):
    """Convert a single frame path to a Nuke %05d sequence pattern.
    
    e.g. '/path/to/input.00130.exr' -> '/path/to/input.%05d.exr'
    """
    dirname = os.path.dirname(file_path)
    basename = os.path.basename(file_path)

    # Match a frame number pattern: digits before the extension
    # e.g. "input.00130.exr" -> groups: ("input.", "00130", ".exr")
    match = re.match(r'^(.*?)(\d+)(\.[^.]+)$', basename)
    if match:
        prefix = match.group(1)
        frame_digits = match.group(2)
        ext = match.group(3)
        padding = len(frame_digits)
        seq_name = "%s%%0%dd%s" % (prefix, padding, ext)
        return os.path.join(dirname, seq_name).replace('\\', '/')

    # Fallback: return the original path if no frame number found
    return file_path
