import re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    def replacement_func(m):
        prefix = m.group(1)
        param_t = m.group(2)
        tcl_string = m.group(3)
        suffix = m.group(4)
        
        inner = tcl_string[1:-1]
        
        # Unescape \n, \", and \\
        inner = inner.replace('\\n', '\n')
        inner = inner.replace('\\"', '"')
        inner = inner.replace('\\\\', '\\')
        
        # In Tcl, if we use curly braces, we must return {inner}
        return f"{prefix}{param_t}{{\n{inner}\n}}{suffix}"

    buttons = ['processCurrentFrameButton', 'updateButton', 'setupEnvButton']
    
    for btn in buttons:
        # Match from `addUserKnob {22 btn_name ` up to `T "..."`
        # Because we're not using greedy wildcards, we can parse just the T part
        pattern = re.compile(r'(addUserKnob \{22 ' + btn + r'.*?)(T )(".*?")(.*?\})', re.DOTALL)
        content = re.sub(pattern, replacement_func, content)

    with open(filepath, 'w') as f:
        f.write(content)

process_file('/Users/josiahvaughan/.nuke/CorridorKeyNuke/CorridorKey.gizmo')
print("Done")
