"""
Global state tracker for CorridorKey Nuke subprocesses.
This allows the UI (such as a 'Cancel' button) to securely terminate a background ML process based on the node name.
"""

# Maps node.fullName() -> active subprocess.Popen instance
active_processes = {}

# Maps node.fullName() -> bool (True if cancellation was requested)
cancel_flags = {}
