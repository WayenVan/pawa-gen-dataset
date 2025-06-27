from transformers import modeling_utils
import json

if (
    not hasattr(modeling_utils, "ALL_PARALLEL_STYLES")
    or modeling_utils.ALL_PARALLEL_STYLES is None
):
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]
import datasets
from datasets import load_dataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from accelerate.utils import DataLoaderConfiguration
from tqdm import tqdm
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
mname = "google/gemma-3-27b-it"

working_dir = "outputs"
output_dir = os.path.join(working_dir, "xlam60k2swahali")
output_file = os.path.join(output_dir, f"output.json")

system = {
    "role": "system",
    "content": """You are a professional translator model that translates English to Swahili.
You should only show translation, do not genearte any function calls or tool calls. Do not add any additional prefixes or suffixes to the translation. The output should only inlucde swahili. You should keep the details of the original query as much as possible, and do not change the meaning of the query.""",
}
rank = int(os.environ.get("LOCAL_RANK", 0))

os.makedirs(output_dir, exist_ok=True)
# ------------------- start distibuted-------------------

# device = torch.device(f"cuda:{rank}")
# dist.init_process_group("nccl", device_id=device)


# ------------------- loading the dataset -------------------

xlam_function60k = load_dataset("Salesforce/xlam-function-calling-60k")
train_subset = xlam_function60k["train"]
loader = DataLoader(train_subset, batch_size=2, shuffle=False)

# ------------------- loading the model -------------------
#
model = AutoModelForCausalLM.from_pretrained(
    mname,
    torch_dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(mname)
translator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=acc.device,
    # device_map="0,1,2,3,4,5,6,7",  # Adjust this based on your GPU setup
    max_length=512,
)
print("----------------beginning of chat template----------------")
print(tokenizer.chat_template)
print("----------------end of chat template----------------")


def do_translate(batch):
    prompts = []
    for q in batch["query"]:
        query = [
            system,
            {
                "role": "user",
                "content": "translate this to swahili and only show the swahili translation: \n"
                + q
                + " /no_think\n",
            },
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                query,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    outputs = translator(
        prompts,
        return_full_text=False,
        do_sample=False,
        num_beams=12,
        temperature=None,
        top_p=None,
        top_k=None,
        # use_cache=False,
    )
    return outputs


# ------------------- preparing the loader -------------------
# loader = acc.prepare(loader)
# if hasattr(loader.batch_sampler, "set_epoch"):

if rank == 0:
    f = open(output_file, "w")
    f.write("[\n")
for batch in tqdm(loader):
    outputs = do_translate(batch)
    if rank == 0:
        for i in range(len(outputs)):
            jobj = {
                "id": batch["id"][i].item(),
                "query_en": batch["query"][i],
                "query_sw": outputs[i][0]["generated_text"],
                "answers": batch["answers"][i],
                "tools": batch["tools"][i],
            }
            f.write(json.dumps(jobj, ensure_ascii=False, indent=4) + ",\n")

if rank == 0:
    f.seek(f.tell() - 2, os.SEEK_SET)  # Remove the last comma
    f.write("\n]\n")
