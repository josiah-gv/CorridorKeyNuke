# Agent Execution Plan: CorridorKey Nuke Integration

This document outlines the step-by-step development process for creating the CorridorKey Nuke Gizmo. It is designed to be clean, simple, and easy for other agents to follow. To ensure stability, **the agent MUST pause execution at the end of each step and wait for the user to manually verify the software in the Nuke GUI** before moving on to the next step.

> **⚠️ AUTO-DEPLOY INSTRUCTION FOR AGENTS:**
> Whenever the agent modifies the `CorridorKey.gizmo` file, it MUST automatically copy the updated file to the user's `~/.nuke/gizmos/` directory so the user does not have to manually move the file for testing.

*Reference Materials:*
- **`docs/gizmo_plan.md`**: The central strategy for features, installation, and cross-platform Nuke versioning.
- **`docs/doc_sources.md`**: The official Nuke 16 Python and UI documentation.
- Nuke 16 Official Python Documentation
- Nuke Gizmo and UI Creation Guides
- Python Native Libraries (`subprocess`, `os`, `sys` for Python 3.10/3.11)
- CorridorKey GitHub Repository

---

## Step 1: Build the Front-End (Gizmo UI)
**Goal:** Construct the base Nuke Gizmo structure with all necessary properties/knobs and ensure it loads into Nuke correctly without errors. No underlying logic is needed yet.
1. Create a base `.gizmo` file (Targeting Nuke 16.0v8 / Python 3.10+ / PySide6.5).
2. Add the node inputs: `Plate` (RGB) and `AlphaHint` (Alpha).
3. Build out the User Interface (Gizmo Controls) following standard Nuke UI guidelines:
   - **Main Tab:** Process Button, Working Directory selector, Frame Range settings, Status text block.
   - **Outputs Tab:** Checkboxes to enable `Matte` and `FG Color` output pipes (Default output is `RGBA`).
   - **CorridorKey Settings Tab:** Gamma Space dropdown (Linear/sRGB), Despill Strength slider, Auto-Despeckle checkbox + size setting, Refiner Strength multiplier.
   - **Paths & Install Tab:** "Install CorridorKey" button, Install Path selector, Custom Python Path selector, Custom Script Path selector.
4. Verify the UI renders correctly in Nuke and all interface elements are present.

> **🛑 USER CHECKPOINT:**
> The agent must pause and notify the user to manually open Nuke, load the newly generated `.gizmo` file, and verify that the custom UI renders correctly with all tabs, inputs, and settings present. Once the user confirms the GUI is correct, proceed to Step 2.

---

## Step 2: Implement the Install Tab Functionality
**Goal:** Get the "plug-and-play" auto-install and auto-detect logic working perfectly using a Hybrid Bootstrap approach.
1. Write the Python backend for the "Install CorridorKey" button action. It should:
   - Run in the background (prevent UI freezing).
   - Have a lightweight "Bootstrap" script embedded directly in the gizmo's PyScript knob.
   - The Bootstrap script clones the CorridorKey GitHub repository into `~/.nuke/CorridorKeyNuke`.
   - The Bootstrap script then dynamically adds `~/.nuke/CorridorKeyNuke` to the Nuke `sys.path` and imports a dedicated installation `.py` module from the repo.
   - The dedicated `.py` module from the repo takes over to create a Python virtual environment (`venv`), install dependencies from `requirements.txt`, and download the core `CorridorKey_v1.0.pth` model to the `/checkpoints` folder.
2. Implement the Auto-Detection logic for Gizmo initialization:
   - When the Gizmo loads, check if `~/.nuke/CorridorKeyNuke` exists.
   - If present, automatically fill in the "CorridorKey Install Path" knob.
   - Utilize `os` and `sys.platform` to ensure cross-platform path formatting (handling Windows `\\` vs Mac/Linux `/` and executable paths like `python.exe` vs `python`).

> **🛑 USER CHECKPOINT:**
> The agent must provide the Python scripts and pause. The user must click the "Install CorridorKey" button in Nuke and verify that the background tasks complete correctly (cloning repo to `~/.nuke/CorridorKeyNuke`, creating venv, downloading the model). The user must also verify auto-detection works upon reloading the node. Once confirmed, proceed to Step 3.

---

## Step 3: Build Core Processing Systems (Rest of Tabs)
**Goal:** Write the main execution logic that bridges Nuke with the CorridorKey machine learning engine.
1. **Pre-processing (Export):**
   - Bind logic to the "Process" button.
   - Create hidden `Write` nodes connected to `Plate` and `AlphaHint`.
   - Render the specified frame range into temporary `/inputs/` staging folders within the defined Working Directory.
2. **Execution (Machine Learning):**
   - Construct a command-line arguments string using values from the UI (Gamma, Despill, Paths, etc.).
   - Use Python's `subprocess.run()` (using Nuke's built-in progress bar and proper `shell` flags based on OS) to launch the ML task in the background.
3. **Post-processing (Import):**
   - Wait for the subprocess to complete.
   - Scan the `/outputs/` directories (`/Matte`, `/FG`, `/Processed`).
   - Automatically generate `Read` nodes for the resulting image sequences.
   - Connect these `Read` nodes to the respective Gizmo outputs (`RGBA`, `Matte`, `FG Color`), toggling visibility based on the "Outputs Tab" checkbox states.

> **🛑 USER CHECKPOINT:**
> The agent must pause. The user will review the execution logic for correctness and safety, and confirm that the Python logic ties cleanly to the Gizmo structure. Once confirmed, proceed to Step 4.

---

## Step 4: Real-World Testing and Validation
**Goal:** Verify the complete functionality on a simple composition.
1. Load a simple image sequence into Nuke.
2. Connect it to the Gizmo along with a basic Alpha Hint.
3. Verify that the intermediate directories (`/inputs/` and `/outputs/`) are built with proper path formatting.
4. Verify that the correct CLI string is generated and `subprocess` fires accurately without freezing Nuke.
5. Verify that the final `Read` nodes are dynamically loaded upon script completion and mapped into the Node Graph correctly.

> **🛑 USER CHECKPOINT:**
> The user must perform a full manual test on real footage inside Nuke. The user must report any Nuke console errors, Python tracebacks, or UI failures to the agent for debugging. The agent will iteratively fix any errors the user provides until the workflow succeeds end-to-end.
