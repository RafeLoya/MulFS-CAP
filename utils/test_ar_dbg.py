import os
import torch

print("[INFO] dir check")
print(f"  current directory:        {os.getcwd()}")
print(f"  VIS test exists:          {os.path.exists('./data/test/vis')}")
print(f"  IR test exists:           {os.path.exists('./data/test/ir')}")
print(f"  results directory exists: {os.path.exists('./results')}")

pretrain_dir = r"./pretrain"
print("\n[INFO] model check")
print(f"  pretrain directory exists: {os.path.exists(pretrain_dir)}")
if os.path.exists(pretrain_dir):
    print(f"    pretrain directory exists: {os.path.exists(pretrain_dir)}")
    model_path = os.path.join(pretrain_dir, "ckpts.pth")
    print(f"    model file exists: {os.path.exists(model_path)}")
    if os.path.exists(model_path):
        try:
            checkpoints = torch.load(model_path)
            print(f"  [INFO] model loaded successfully")
            print(f"  [INFO] model keys: {checkpoints.keys()}")
        except Exception as e:
            print(f"  [ERROR] failed to load model: {e}")
            
print("\n[INFO] CUDA check")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  [INFO] GPU: {torch.cuda.get_device_name(0)}")
