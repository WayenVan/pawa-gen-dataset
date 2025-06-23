import datasets
from datasets import load_dataset
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate.utils import DataLoaderConfiguration
import torch

data_loader_config = DataLoaderConfiguration(
    even_batches=False,
    use_seedable_sampler=True,
)

acc = Accelerator(mixed_precision="no", dataloader_config=data_loader_config)

working_dir = "outputs"
model_name = "jbochi/madlad400-3b-mt"
output_dir = os.path.join(working_dir, "xlam60k2swahali")
output_file = os.path.join(output_dir, f"output_rank{acc.local_process_index}.json")


os.makedirs(output_dir, exist_ok=True)
# ------------------- loading the dataset -------------------

xlam_function60k = load_dataset("Salesforce/xlam-function-calling-60k")
train_subset = xlam_function60k["train"].select(range(10))
loader = DataLoader(train_subset, batch_size=2, shuffle=False)

# ------------------- loading the model -------------------
#
translator = T5ForConditionalGeneration.from_pretrained(
    model_name, device_map="cpu", torch_dtype=torch.float32
).to(acc.device)
tokenizer = T5Tokenizer.from_pretrained(model_name)
loader = acc.prepare(loader)

# if hasattr(loader.batch_sampler, "set_epoch"):

with open(output_file, "w") as f:
    for batch in loader:
        query = ["<2sw> " + q for q in batch["query"]]

        if acc.is_main_process:
            print(query)

        input_ids = tokenizer(query, padding=True, return_tensors="pt").input_ids.to(
            acc.device
        )

        outputs = translator.generate(input_ids=input_ids, num_beams=10)

        out = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if acc.is_main_process:
            print(out)
