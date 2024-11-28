from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
#model = Qwen2VLForConditionalGeneration.from_pretrained(
#    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
#)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
     "Qwen/Qwen2-VL-7B-Instruct",
     torch_dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
#min_pixels = 256*28*28
#max_pixels = 1280*28*28
#processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

#messages = [
#    {
#        "role": "user",
#        "content": [
#            {
#                "type": "image",
#                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#            },
#            {"type": "text", "text": "Describe this image."},
#        ],
#    }
#]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///home/ozdomar/Adil/doc_gpt/vision/data/test-new4.png"},
            {"type": "text","text": "Please read the table from the provided image and transcribe it into JSON format. Each row of the table should be represented as a JSON object with clear keys for each column. Ensure that all details are accurately captured and the JSON is well-formatted."},
#            {"type": "text", "text": "Please read the table from the provided image and transcribe it as structured text. Format the output in a tabular style using plain text, where each row is represented in a structured format. Use clear column names and align the values under their respective headers. Ensure that the output captures all details accurately as they appear in the table."},
        ],
    }
]


# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
