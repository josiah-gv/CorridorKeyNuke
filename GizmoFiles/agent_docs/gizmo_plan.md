# CorridorKey Nuke Integration - Development Plan

## Goal Description
Develop a simple, efficient Nuke Gizmo that acts as a bridge to the CorridorKey machine learning greenscreen tool. The Gizmo will allow users to input a greenscreen plate and an alpha hint, run the CorridorKey inference process from within Nuke, and automatically load the generated matte and color passes back into the comp. 

## Installation Strategy: Auto-Install & Auto-Detect (Hybrid Bootstrap)
To provide a seamless "plug-and-play" experience, the Gizmo will handle the installation and configuration of CorridorKey automatically, while maintaining clean separation of concerns.

- **One-Click Installation (The Hybrid Bootstrap):** The Gizmo will include an "Install CorridorKey" button. When clicked:
  1. **Phase 1 (The Clone):** A lightweight bootstrap python script embedded directly in the gizmo downloads the repository to `~/.nuke/CorridorKeyNuke`.
  2. **Phase 2 (The Hand-off):** The bootstrap script dynamically adds `~/.nuke/CorridorKeyNuke` to the Nuke Python `sys.path` and imports a dedicated installation `.py` module directly from the newly cloned repo.
  3. **Phase 3 (The Heavy Lifting):** The external `.py` module runs securely in the background to create a dedicated Python virtual environment (`venv`), install requirements, and download the core `CorridorKey_v1.0.pth` model to the `/checkpoints` folder.
- **Auto-Detection:** Every time the Gizmo is loaded, it will check if `~/.nuke/CorridorKeyNuke` (and the required model) exists. If found, it will automatically populate the "Install Path" internally so the user can start using the tool immediately without manual configuration.

## Compatibility

### Nuke Versions
The Gizmo will be explicitly developed and targeted for **Nuke 16.0v8**.
- **Python Backend:** Nuke 16 uses Python 3.10 natively, which aligns perfectly with CorridorKey's own Python 3.10 requirement. By using standard Nuke `nuke.execute()` and standard Python `subprocess` modules, the workflow will be highly stable within this version.

### Cross-Platform (Windows, Mac, Linux)
**We only need one Gizmo and one Python script.** 
A single Nuke integration can dynamically support all operating systems because Python (which drives the Gizmo's logic) has built-in OS detection capabilities. 

Here is how the tool will handle OS differences under the hood:
- **Path formatting:** Windows uses backslashes (`\`) for file paths, while Mac/Linux use forward slashes (`/`). The Python backend will use `os.path.join` and `os.path.normpath` to ensure that paths are formatted correctly for the specific system running Nuke.
- **Python Executable location:** In normal Python virtual environments (`venv`), the path to the executable differs:
  - **Windows:** `venv\Scripts\python.exe`
  - **Mac/Linux:** `venv/bin/python`
  - The Gizmo's python script will check `sys.platform` to automatically deduce the correct path to the CorridorKey python executable if the user leaves the "Custom Python Path" blank.
- **Subprocess Execution:** When calling `subprocess.run()`, Windows sometimes requires `shell=True` to find executables properly, while Mac/Linux prefer `shell=False`. The python script will adapt this flag based on the detected OS.

## Feature List

### 1. Node Inputs
- `Plate`: The base greenscreen footage (RGB).
- `AlphaHint`: A rough black-and-white mask (Alpha) generated via Keylight, Primatte, or roto.

### 2. Node Outputs
By default, only the standard `RGBA` output is visible to keep the Node Graph clean. Additional outputs can be enabled in the Gizmo's properties.
- `RGBA` (Default): The pre-multiplied final composite (Processed pass).
- `Matte` (Optional): The raw linear Alpha channel.
- `FG Color` (Optional): The raw un-premultiplied foreground color.

### 3. User Interface (Gizmo Controls)
- **Main Tab:**
  - **Process Button:** Triggers the rendering and ML inference.
  - **Working Directory:** Automatically defaults to `[file dirname [value root.name]]/CorridorKey/[value name]`. Subfolders `/inputs` and `/outputs` will be created here.
  - **Frame Range:** Start and End frames to process.
  - **Status Text:** Displays current script status (Extracing frames, Processing ML, Loading Output).
- **Outputs Tab:**
  - **Enable Matte Output:** Checkbox to reveal the Matte output pipe.
  - **Enable FG Color Output:** Checkbox to reveal the FG Color output pipe.
- **CorridorKey Settings Tab:**
  - **Gamma Space:** Linear or sRGB.
  - **Despill Strength:** Slider for despill (0-10).
  - **Auto-Despeckle:** Checkbox and size threshold.
  - **Refiner Strength:** Multiplier (default 1.0).
- **Paths & Install Tab:**
  - **Install CorridorKey Button:** Downloads the repo via embedded bootstrap script, hands off to the repo's python module to build the `venv`, and downloads the core ML model to `~/.nuke/CorridorKeyNuke`.
  - **CorridorKey Install Path:** The root directory where CorridorKey is installed. This auto-populates to `~/.nuke/CorridorKeyNuke` if detected. Can be overridden manually.
  - **Custom Python Path (Optional):** Path to the python executable in the CorridorKey virtual environment.
  - **Custom Script Path (Optional):** Path to the launcher script.

## Functionality Plan: Under the Hood

When the user clicks the "Process" button, a Python script will execute the following sequence:

1. **Pre-processing (Export):**
   - Nuke creates hidden `Write` nodes internally connected to the `Plate` and `AlphaHint`.
   - It renders the specified frame range as image sequences into temporary staging folders inside the defined Working Directory (e.g., `.../inputs/Input/` and `.../inputs/AlphaHint/`).

2. **Execution (The Machine Learning):**
   - The script builds a command-line string pointing to the CorridorKey environment using the user's settings (Gamma, Despill, etc.).
   - Nuke uses Python's `subprocess` module to run CorridorKey in the background. Nuke's GUI will show a progress bar waiting for the subprocess to complete.

3. **Post-processing (Import):**
   - Once CorridorKey finishes writing files to its designated `/Matte`, `/FG`, and `/Processed` folders (which we'll configure CorridorKey to output into `.../outputs/`), the Nuke script scans these directories.
   - It automatically creates `Read` nodes inside the Gizmo for each corresponding sequence.
   - These `Read` nodes are internally connected to the Gizmo's outputs, seamlessly updating the node tree in the user's Node Graph.

## Next Steps for Development (Post-Approval)
1. Write the Python backend that handles exporting frames, calling the `subprocess`, and importing the results.
2. Group the required nodes (Inputs, Output connections) in Nuke and export them as a `.gizmo`.
3. Link the Python backend to the Gizmo's buttons.
   
## Verification Plan
### Manual Verification
1. We will simulate running the Gizmo on a simple image script to verify that:
   - Output paths construct correctly based on the Nuke script's location.
   - The pre-processing step correctly writes frames to disk in the separated input directories.
   - The subprocess call constructs the correct CLI arguments based on the UI settings and default paths.
   - Read nodes are dynamically populated upon completion, and optional outputs toggle correctly.
