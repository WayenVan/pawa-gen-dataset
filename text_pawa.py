from transformers import modeling_utils

if (
    not hasattr(modeling_utils, "ALL_PARALLEL_STYLES")
    or modeling_utils.ALL_PARALLEL_STYLES is None
):
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "sartifyllc/pawa-min-beta", torch_dtype="bfloat16"
)
tokenizer = AutoTokenizer.from_pretrained("sartifyllc/pawa-min-beta")

messages = [
    {"role": "system", "content": "You are a translator model."},
    {"role": "user", "content": "Who are you?"},
]
tmp = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(tmp)

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device=0,
# )
# print(pipe(messages))
