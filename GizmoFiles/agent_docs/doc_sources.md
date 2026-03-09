# Documentation Sources - CorridorKey Nuke Integration

To ensure the Nuke Gizmo is built to the proper specifications using modern practices and avoids reliance on outdated forum posts, the following official documentation sources will be used as the primary references during development:

## 1. Nuke 16 Official Python Documentation
Nuke 16 introduces specific updates to the Python API (transitioning to Python 3.11 and Qt/PySide 6.5). We will adhere to these official Foundry resources:

*   **Nuke Python Developer's Guide (Nuke 16.0)**
    *   *Purpose:* Conceptual overview of using Python within Nuke, best practices for executing background scripts, and managing the node graph.
    *   *Link:* [Nuke Python Developer's Guide](https://learn.foundry.com/nuke/developers/16.0/pythondevguide/)
*   **Python API Reference for Nuke 16.0v8**
    *   *Purpose:* The exact namespace, classes, and methods available for building node connections and extracting property values in Nuke 16.0v8.
    *   *Link:* [Nuke 16.0 Python Reference](https://learn.foundry.com/nuke/developers/16.0/pythonreference/)

## 2. Nuke Gizmo and UI Creation
For designing the user-facing properties panel and packaging the script into a `.gizmo` file:

*   **Customizing the UI (Python Dev Guide)**
    *   *Purpose:* Best practices for adding custom tabs, knobs (buttons, path selectors, checkboxes), and linking python callback scripts to the "Process" button.
    *   *Link:* [Customizing the UI](https://learn.foundry.com/nuke/developers/16.0/pythondevguide/custom_ui.html)
*   **Creating User Plugins (Gizmos)**
    *   *Purpose:* Official workflow for exporting a node network structure with promoted knobs into a reusable Gizmo file.
    *   *Link:* [Creating Gizmos (Foundry Learn)](https://learn.foundry.com/nuke/content/comp_environment/organizing_scripts/creating_gizmos.html)

## 3. CorridorKey ML Engine
For executing the machine learning engine faithfully and interpreting its inputs/outputs:

*   **CorridorKey GitHub Repository**
    *   *Purpose:* The core documentation detailing the required file structure (e.g., `/inputs/`, `/outputs/`), specific CLI flags (Gamma Space, Despill), and python environment requirements.
    *   *Link:* [CorridorKey by nikopueringer](https://github.com/nikopueringer/CorridorKey)

## 4. Python Native Libraries (v3.10 / v3.11)
Since the Gizmo acts as a bridge firing off external scripts, standard Python library documentation is critical for cross-platform stability:

*   **`subprocess` module**
    *   *Purpose:* Safely launching the CorridorKey engine in the background without freezing the Nuke UI, while capturing progress or errors. 
    *   *Link:* [Python subprocess documentation](https://docs.python.org/3/library/subprocess.html)
*   **`os` and `sys` modules**
    *   *Purpose:* Handling cross-platform file paths and determining execution contexts across Windows, Mac, and Linux seamlessly.
    *   *Link:* [Python os path documentation](https://docs.python.org/3/library/os.path.html)
