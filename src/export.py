import click
import torch
import onnx
import onnxruntime as ort

from src.lightning_module import OCRModule


@click.command()
@click.argument('model_ckpt_path')
def main(model_ckpt_path: str):
    pl_model = OCRModule.load_from_checkpoint(model_ckpt_path, map_location='cpu')
    crnn_model = pl_model._model

    crnn_model = crnn_model.to('cpu')
    crnn_model.eval()

    with torch.no_grad():
        crnn_ts_model = torch.jit.script(crnn_model)

    # Export TorchScript model to ONNX
    dummy_input = torch.randn((1, 3, 96, 416))
    onnx_model_path = model_ckpt_path.replace('.ckpt', '.onnx')

    torch.onnx.export(
        crnn_model, # crnn_ts_model
        dummy_input,
        onnx_model_path,
        input_names=['input'],
        output_names=['output'],
    )

    onnx.checker.check_model(onnx_model_path)
    print('ONNX checked model')

    ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    onnx_pred = torch.from_numpy(ort_session.run(None, {'input': dummy_input.detach().numpy()})[0])
    ts_pred = crnn_ts_model(dummy_input)
    torch_pred = crnn_model(dummy_input)

    ts_torch_diff = torch.max(torch.abs(ts_pred - torch_pred))
    onnx_torch_diff = torch.max(torch.abs(onnx_pred - torch_pred))
    onnx_ts_diff = torch.max(torch.abs(onnx_pred - ts_pred))

    print('TS_TORCH_MAX_DIFF: ', ts_torch_diff)
    print('ONNX_TORCH_MAX_DIFF: ', onnx_torch_diff)
    print('ONNX_TS_MAX_DIFF: ', onnx_ts_diff)


if __name__ == '__main__':
    main()
