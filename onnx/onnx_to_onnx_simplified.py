import onnxsim
import onnx

simplified_onnx_model, success = onnxsim.simplify('model.onnx')
assert success, 'Failed to simplify the ONNX model. You may have to skip this step'
simplified_onnx_model_path = f'model.simplified.onnx'

onnx.save(simplified_onnx_model, simplified_onnx_model_path)
print('Model has been simplified')
