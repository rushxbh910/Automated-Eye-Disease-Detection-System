import os
import sys
import onnx
import onnxruntime
import time
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def optimize_and_quantize_onnx_model(onnx_model_path: str, export_dir: str = "Artifacts/serving_models"):
    try:
        logging.info("Starting ONNX model optimization and quantization")

        optimized_model_path = os.path.join(export_dir, "model_optimized.onnx")
        quantized_model_path = os.path.join(export_dir, "model_quantized.onnx")

        # Quantize model dynamically to reduce size and improve inference speed
        quantize_dynamic(
            model_input=onnx_model_path,
            model_output=quantized_model_path,
            weight_type=QuantType.QUInt8
         
        )
        logging.info(f"Quantized model saved at: {quantized_model_path}")

        # Benchmark inference performance on quantized model
        session = onnxruntime.InferenceSession(quantized_model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        dummy_input = torch.randn(1, 3, 224, 224).numpy()

        warmup_runs = 10
        benchmark_runs = 100
        logging.info("Warming up...")
        for _ in range(warmup_runs):
            _ = session.run(None, {input_name: dummy_input})

        logging.info("Starting benchmark...")
        start = time.time()
        for _ in range(benchmark_runs):
            _ = session.run(None, {input_name: dummy_input})
        end = time.time()

        avg_latency_ms = (end - start) / benchmark_runs * 1000
        throughput_fps = 1000 / avg_latency_ms

        logging.info(f"✅ Average latency: {avg_latency_ms:.2f} ms")
        logging.info(f"🚀 Throughput: {throughput_fps:.2f} FPS")

    except Exception as e:
        raise NetworkSecurityException(e, sys)



