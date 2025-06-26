from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from vllm.inputs.data import TextPrompt
from transformers import modeling_utils
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

if (
    not hasattr(modeling_utils, "ALL_PARALLEL_STYLES")
    or modeling_utils.ALL_PARALLEL_STYLES is None
):
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

import datasets
from datasets import load_dataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["VLLM_USE_V1"] = "0"  # Use v1 API for vLLM
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
mname = "google/gemma-3-27b-it"
# mname = "sartifyllc/pawa-min-alpha"

working_dir = "outputs"
output_dir = os.path.join(working_dir, "xlam60k2swahali")
output_file = os.path.join(output_dir, f"output.json")

system = {
    "role": "system",
    "content": """You are a professional translator model that translates English to Swahili.
You should only show translation, do not genearte any function calls or tool calls. Do not add any additional prefixes or suffixes to the translation. The output should only inlucde swahili. You should keep the details of the original query as much as possible, and do not change the meaning of the query.""",
}


os.makedirs(output_dir, exist_ok=True)
# ------------------- loading the dataset -------------------

xlam_function60k = load_dataset("Salesforce/xlam-function-calling-60k")
train_subset = xlam_function60k["train"]
loader = DataLoader(train_subset, batch_size=2, shuffle=False)

# ------------------- loading the model -------------------
model = LLM(
    model=mname,
    task="generate",
    model_impl="vllm",
    tensor_parallel_size=8,  # Adjust based on your GPU setup
    enforce_eager=True,
    dtype="bfloat16",
    enable_prefix_caching=True,
)

# ------------------- loading the tokenizer -------------------
tokenizer = AutoTokenizer.from_pretrained(mname)
print("----------------beginning of chat template----------------")
print(tokenizer.chat_template)
print("----------------end of chat template----------------")


# ------------------- preparing the processor -------------------
def do_translate(batch):
    prompts = []
    for q in batch["query"]:
        query = [
            system,
            {
                "role": "user",
                "content": "translate this to swahili and only show the swahili translation:  \n"
                + q
                + " /no_think \n",
            },
        ]
        prompts.append(
            dict(
                prompt=tokenizer.apply_chat_template(
                    query,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        )

    # beam_params = BeamSearchParams(
    #     beam_width=10,
    #     max_tokens=512,
    # )
    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=1024)
    outputs = model.generate(
        prompts,
        sampling_params=sampling_params,
    )
    # outputs = model.beam_search(
    #     prompts,
    #     params=beam_params,
    # )
    return outputs


with open(output_file, "w") as f:
    f.write("[\n")
    for batch in tqdm(loader):
        outputs = do_translate(batch)
        for i in range(len(outputs)):
            jobj = {
                "id": batch["id"][i].item(),
                "query_en": batch["query"][i],
                "query_sw": outputs[i].outputs[0].text,
                # "query_sw": outputs[i].sequences[0].text,
                "answers": json.loads(batch["answers"][i]),
                "tools": json.loads(batch["tools"][i]),
            }
            f.write(json.dumps(jobj, ensure_ascii=False, indent=4) + ",\n")

    f.seek(f.tell() - 2, os.SEEK_SET)  # Remove the last comma
    f.write("\n]\n")
