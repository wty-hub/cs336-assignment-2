"""Reproduce an FP16-autocast Linear forward on the current CUDA GPU."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['trace', 'ncu'], default='trace')
args = parser.parse_args()
out = Path(__file__).resolve().parent
torch.manual_seed(0)
model = nn.Linear(768, 3072, bias=True, device='cuda')
x = torch.randn(1024, 768, device='cuda')

def forward():
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        return model(x)

for _ in range(5):
    forward()
torch.cuda.synchronize()

if args.mode == 'trace':
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as p:
        y = forward()
        torch.cuda.synchronize()
    p.export_chrome_trace(str(out / 'trace.json'))
    trace = json.loads((out / 'trace.json').read_text())
    kernels = [e['name'] for e in trace['traceEvents'] if e.get('cat') == 'kernel']
    evidence = {
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'input_shape': list(x.shape), 'weight_shape': list(model.weight.shape),
        'parameter_dtype': str(model.weight.dtype), 'input_dtype': str(x.dtype),
        'autocast_dtype': 'torch.float16', 'output_dtype': str(y.dtype),
        'allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'allow_fp16_accumulation': torch.backends.cuda.matmul.allow_fp16_accumulation,
        'kernels': kernels,
        'loaded_cublas_libraries': sorted({line.split()[-1] for line in open('/proc/self/maps') if 'libcublas' in line}),
    }
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))
else:
    torch.cuda.cudart().cudaProfilerStart()
    y = forward()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print('output dtype:', y.dtype)
