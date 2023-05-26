# Ruhtils
## Utils functions for faster ML programming

#### How to measure FLOPS
```
from fvcore.nn import FlopCountAnalysis
import torch


model = torch.load("model.pt")
flops = FlopCountAnalysis(model, torch.rand(1,3,640,640))
flops.total()
```
