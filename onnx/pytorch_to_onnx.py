import torch.jit
from models import Generator

model = Generator()
model.load_state_dict(torch.load('initial_model_checkpoint.pth', map_location=torch.device('cpu')))

output_onnx_file = 'onnx/model.onnx'
dummy_input = torch.randn(1, 3, 540, 960)

torch.onnx.export(model,
                  dummy_input,
                  output_onnx_file,
                  opset_version=12,
                  verbose=True,
                  input_names=['input'],
                  output_names=['output'])
