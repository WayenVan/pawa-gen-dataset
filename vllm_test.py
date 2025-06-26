from vllm import LLM, SamplingParams
import os
from transformers import modeling_utils

if (
    not hasattr(modeling_utils, "ALL_PARALLEL_STYLES")
    or modeling_utils.ALL_PARALLEL_STYLES is None
):
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

os.environ["VLLM_USE_V1"] = "0"  # Use v1 API for vLLM
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # Use v1 API for vLLM
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def test_model(model_id: str):
    try:
        llm = LLM(
            model=model_id,
            task="generate",
            model_impl="vllm",
            tensor_parallel_size=2,  # Adjust based on your GPU setup
        )
        prompts = ["你好，vLLM！", "今天天气如何？"]
        # sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
        outputs = llm.generate(prompts)
        for out in outputs:
            print(f"Prompt: {out.prompt!r}")
            print(f"Generated: {out.outputs[0].text!r}")
            print("-" * 40)
        print(f"✅ 模型 {model_id} 支持并运行成功！")
    except Exception as e:
        print(f"❌ 模型 {model_id} 加载失败：{e}")


if __name__ == "__main__":
    # 可替换为你想测试的模型 ID
    # test_model("google/gemma-3-4b-it")
    # test_model("google/gemma-3-4b-it")
    test_model("google/gemma-3-4b-it")
    # test_model("Qwen/Qwen3-32B")
    # test_model("your-model-id")
