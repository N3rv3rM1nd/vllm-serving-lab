import argparse, subprocess, sys
from pathlib import Path

STEPS = ["result_Bn_check", "engine_metric", "merge_result",
         "goodput_Bn"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline_n256")
    ap.add_argument("--only", nargs="*", help="只跑指定步骤")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent / "analysis"
    for step in (args.only or STEPS):
        print(f">>> {step}", flush=True)
        subprocess.run([sys.executable, str(root / f"{step}.py"),
                        "--label", args.label], check=True)

if __name__ == "__main__":
    main()