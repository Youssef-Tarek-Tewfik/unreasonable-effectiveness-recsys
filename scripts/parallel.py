import subprocess
from pathlib import Path

from source.constants import PATH_SCRIPT_SEQUENTIAL, Tool, Scorer, Model


with open(PATH_SCRIPT_SEQUENTIAL, 'r') as f:
    template = f.read()

tags = [
    # One run per tool-algorithm combination; "[TOOL]:[ALGORITHM]"
    # *[f"{Tool.LENSKIT.name}:{scorer.name}" for scorer in Scorer], 
    # *[f"{Tool.RECBOLE.name}:{model.name}" for model in Model],

    # One run per tool; "[TOOL]:"
    f"{Tool.LENSKIT.name}:", f"{Tool.RECBOLE.name}:"
]
for tag in tags:
    name = f"parallel-{tag.replace(':', '-')}"
    script = template + f" --tag {tag}"
    
    with open(f"./{name}.sh", 'w', newline='\n') as f:
        f.write(script)
    
    subprocess.run(["sbatch", f"./{name}.sh"])
    
    Path(f"./{name}.sh").unlink()
