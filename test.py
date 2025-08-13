import torch
print("=== Device Check ===")
print("CUDA Available:", torch.cuda.is_available())
num_gpus = torch.cuda.device_count()
print("Number of GPUs:", num_gpus)

if torch.cuda.is_available():
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"  Memory Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
print("====================")

print("torch:", torch.version)