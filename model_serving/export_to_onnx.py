import os
import sys
import torch
import traceback
from torchvision.models import resnet
import logging

def export_pytorch_to_onnx(model_path: str = None, export_dir: str = None):
    """
    Converts a trained PyTorch model to ONNX format.

    Args:
        model_path (str): Path to the saved PyTorch model.
        export_dir (str): Directory to save ONNX model.
    """
    try:
        logging.info("🚀 Starting PyTorch to ONNX export...")

        # Load from environment or defaults
        model_path = model_path or os.environ.get("TRAINED_MODEL_PATH", "Artifacts/model.pth")
        export_dir = export_dir or os.environ.get("ONNX_EXPORT_DIR", "Artifacts/serving_models")

        os.makedirs(export_dir, exist_ok=True)
        onnx_path = os.path.join(export_dir, "model.onnx")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Safe deserialization
        try:
            torch.serialization.add_safe_globals({"torchvision.models.resnet.ResNet": resnet.ResNet})
        except Exception as e:
            logging.warning(f"Safe globals registration failed: {e}")

        # Load model (full object)
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224).to(device)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            export_params=True,
            opset_version=11
        )

        logging.info(f"✅ ONNX model successfully exported to: {onnx_path}")

    except Exception as e:
        logging.error(f"❌ ONNX export failed: {traceback.format_exc()}")
        raise Exception(e, sys)
