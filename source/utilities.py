import torch
import os
import pandas as pd
from pathlib import Path

from .constants import DIRECTORY_DATASETS, Dataset
from .load import load


def main():
    pass

def gpu_check() -> None:
    print("=== GPU Detection ===")
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("\n=== GPU Details ===")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-processors: {props.multi_processor_count}")

            # Test GPU accessibility
            try:
                torch.cuda.set_device(i)
                test_tensor = torch.tensor([1.0]).cuda()
                print(f"  Status: ✓ Accessible")
            except Exception as e:
                print(f"  Status: ✗ Error - {e}")

        print(f"\n=== Current GPU ===")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Current device name: {torch.cuda.get_device_name()}")

        print(f"\n=== Environment Variables ===")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    else:
        print("No CUDA GPUs detected")

def inter_to_csv(name: str, columns: list[int], output="ratings", demo=False) -> None:
    input_path = DIRECTORY_DATASETS / name / f"{'demo' if demo else name}.inter"
    output_path = DIRECTORY_DATASETS / name / f"{output}.csv"
    
    df = pd.read_csv(input_path, sep='\t', skiprows=1, header=None)
    
    df = df.iloc[:, columns]
    
    df.to_csv(output_path, index=False, header=False)
    
    print(f"Converted\n{input_path}\nto\n{output_path}")

def remove_last_column(input: str | Path, output: str | Path) -> None:
    df = pd.read_csv(input, header=None)
    df = df.iloc[:, :-1]
    df.to_csv(output, index=False, header=False)
    print(f"Removed last column from\n{input}\nand saved to\n{output}")

def copy_head(input: str | Path, output: str | Path) -> None:
    with open(input, 'r') as input_file:
        lines = [input_file.readline() for _ in range(5)]
    with open(output, 'w') as output_file:
        output_file.writelines(lines)
    print(f"Copied first 5 lines from\n{input}\nto\n{output}")

if __name__ == "__main__":
    main()
