import onnx
import numpy
import onnxruntime


def onnx_layer_output(onnx_path, dummy_input):
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    org_outputs = [x.name for x in ort_session.get_outputs()]
    model = onnx.load('yolov3.onnx')
    for node in model.graph.node:
        for output in node.output:
            if output not in org_outputs:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    # excute onnx
    ort_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=providers)
    outputs = [x.name for x in ort_session.get_outputs()]
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(outputs, ort_inputs)
    ort_outs = OrderedDict(zip(outputs, ort_outs))
    print(ort_outs)