from pathlib import Path
import shutil
import subprocess
import sys
import time
import os
import traceback


def run_cyclegan(content_path, output_path):
    base_dir = Path(__file__).resolve().parent.parent
    repo_dir = base_dir / "pytorch-CycleGAN-and-pix2pix"
    input_root = base_dir / "uploads" / "cyclegan_input"
    testA_dir = input_root / "testA"
    testB_dir = input_root / "testB"
    results_dir = base_dir / "uploads" / "cyclegan_results"
    checkpoints_dir = base_dir / "checkpoints"

    output_path = Path(output_path)
    content_path = Path(content_path)

    print("=" * 60, flush=True)
    print("[CycleGAN] Start inference", flush=True)
    print(f"[CycleGAN] content_path   = {content_path}", flush=True)
    print(f"[CycleGAN] output_path    = {output_path}", flush=True)
    print(f"[CycleGAN] repo_dir       = {repo_dir}", flush=True)
    print(f"[CycleGAN] input_root     = {input_root}", flush=True)
    print(f"[CycleGAN] checkpoints_dir= {checkpoints_dir}", flush=True)
    print(f"[CycleGAN] results_dir    = {results_dir}", flush=True)

    try:
        if not content_path.exists():
            raise FileNotFoundError(f"content image not found: {content_path}")
        if not repo_dir.exists():
            raise FileNotFoundError(f"CycleGAN repo not found: {repo_dir}")
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"checkpoints dir not found: {checkpoints_dir}")

        testA_dir.mkdir(parents=True, exist_ok=True)
        testB_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("[CycleGAN] Clearing old test files...", flush=True)
        for folder in [testA_dir, testB_dir]:
            for file in folder.iterdir():
                if file.is_file():
                    file.unlink()

        print("[CycleGAN] Preparing input files...", flush=True)
        input_name = f"input_{int(time.time())}{content_path.suffix.lower()}"
        target_input_a = testA_dir / input_name
        target_input_b = testB_dir / input_name

        # 为了适配 unaligned 目录结构，这里两边都放同一张内容图
        shutil.copy(content_path, target_input_a)
        shutil.copy(content_path, target_input_b)

        print(f"[CycleGAN] testA input = {target_input_a}", flush=True)
        print(f"[CycleGAN] testB input = {target_input_b}", flush=True)

        test_py = repo_dir / "test.py"
        if not test_py.exists():
            raise FileNotFoundError(f"test.py not found: {test_py}")

        cmd = [
            sys.executable,
            str(test_py),
            "--dataroot", str(input_root),
            "--name", "coco2vango",
            "--model", "cycle_gan",
            "--dataset_mode", "unaligned",
            "--direction", "AtoB",
            "--checkpoints_dir", str(checkpoints_dir),
            "--results_dir", str(results_dir),
            "--epoch", "200",
            "--num_test", "1",
            "--preprocess", "resize_and_crop",
            "--load_size", "286",
            "--crop_size", "256",
        ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        print("[CycleGAN] Running command:", flush=True)
        print(" ".join(cmd), flush=True)

        result = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        print("[CycleGAN] subprocess return code:", result.returncode, flush=True)
        print("[CycleGAN] stdout:", flush=True)
        print(result.stdout, flush=True)
        print("[CycleGAN] stderr:", flush=True)
        print(result.stderr, flush=True)

        if result.returncode != 0:
            raise RuntimeError(
                "CycleGAN 推理失败\n"
                f"stdout:\n{result.stdout}\n\n"
                f"stderr:\n{result.stderr}"
            )

        print("[CycleGAN] Searching output files...", flush=True)
        candidate_files = list(results_dir.rglob("*fake_B*.png")) + list(results_dir.rglob("*fake_B*.jpg"))
        if not candidate_files:
            candidate_files = list(results_dir.rglob("*.png")) + list(results_dir.rglob("*.jpg"))

        if not candidate_files:
            raise FileNotFoundError("未在 CycleGAN 推理结果中找到输出图片")

        latest_file = max(candidate_files, key=lambda p: p.stat().st_mtime)
        print(f"[CycleGAN] latest output file = {latest_file}", flush=True)

        shutil.copy(latest_file, output_path)
        print(f"[CycleGAN] Final output saved: {output_path}", flush=True)
        print("[CycleGAN] Finished successfully.", flush=True)
        print("=" * 60, flush=True)

    except Exception as e:
        print("[CycleGAN] ERROR occurred!", flush=True)
        print(f"[CycleGAN] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise