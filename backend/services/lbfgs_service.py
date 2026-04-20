from pathlib import Path
import shutil


def run_lbfgs(content_path, style_path, output_path):
    content_path = Path(content_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(content_path, output_path)