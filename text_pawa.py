from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


model = AutoModelForCausalLM.from_pretrained(
    "sartifyllc/pawa-min-beta", device_map="cuda:0", torch_dtype="bfloat16"
)
tokenizer = AutoTokenizer.from_pretrained("sartifyllc/pawa-min-beta", device_map="cpu")
pipe = pipeline(
    "text-generation",
    model="sartifyllc/pawa-min-beta",
    tokenizer=tokenizer,
    device=0,
)
messages = [
    {"role": "user", "content": "Who are you?"},
]
print(pipe(messages))

# print(model.__class__.__name__)
