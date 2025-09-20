import torch
import os


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
