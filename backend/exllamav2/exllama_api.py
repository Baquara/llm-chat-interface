
import sys, os, time, math
from fastapi import FastAPI
import torch
import time
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import base64


#pipe = StableDiffusionPipeline.from_single_file("/run/media/privateserver/NECTAC1TBSSD/a1111/stable-diffusion-webui/models/Stable-diffusion/sd15.safetensors", torch_dtype=torch.float16).to("cuda")

def stable_diffusion(prompt):
    # Generate an image using a prompt
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    image = pipe(prompt=prompt, num_inference_steps=30).images[0]
    
    # Save the image directly as PNG
    image.save("generated_image.png")


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()



origins = [
    "http://localhost:3000",  # React app address
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
    model_init,
)

import argparse
import torch

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

from examples.chat_formatting import CodeBlockFormatter
from examples.chat_prompts import prompt_formats
prompt_formats_list = list(prompt_formats.keys())

# Options

parser = argparse.ArgumentParser(description = "Simple Llama2 chat example for ExLlamaV2")
parser.add_argument("-dm", "--draft_model_dir", type = str, default = None, help = "Path to draft model directory")
parser.add_argument("-nds", "--no_draft_scale", action = "store_true", help = "If draft model has smaller context size than model, don't apply alpha (NTK) scaling to extend it")

parser.add_argument("-modes", "--modes", action = "store_true", help = "List available modes and exit.")
parser.add_argument("-mode", "--mode", choices = prompt_formats_list, help = "Chat mode. Use llama for Llama 1/2 chat finetunes.")
parser.add_argument("-un", "--username", type = str, default = "User", help = "Username when using raw chat mode")
parser.add_argument("-bn", "--botname", type = str, default = "Chatbot", help = "Bot name when using raw chat mode")
parser.add_argument("-sp", "--system_prompt", type = str, help = "Use custom system prompt")

parser.add_argument("-temp", "--temperature", type = float, default = 0.70, help = "Sampler temperature, default = 0.95 (1 to disable)")
parser.add_argument("-topk", "--top_k", type = int, default = 20, help = "Sampler top-K, default = 20 (0 to disable)")
parser.add_argument("-topp", "--top_p", type = float, default = 0.9, help = "Sampler top-P, default = 0.9 (0 to disable)")
parser.add_argument("-typical", "--typical", type = float, default = 1.0, help = "Sampler typical threshold, default = 1.0 (0 to disable)")
parser.add_argument("-repp", "--repetition_penalty", type = float, default = 1.15, help = "Sampler repetition penalty, default = 1.15 (1 to disable)")
parser.add_argument("-maxr", "--max_response_tokens", type = int, default = 4096, help = "Max tokens per response, default = 1000")
parser.add_argument("-resc", "--response_chunk", type = int, default = 250, help = "Space to reserve in context for reply, default = 250")
parser.add_argument("-ncf", "--no_code_formatting", action = "store_true", help = "Disable code formatting/syntax highlighting")

parser.add_argument("-c8", "--cache_8bit", action = "store_true", default = True, help = "Use 8-bit cache")

parser.add_argument("-pt", "--print_timings", action = "store_true", help = "Output timings after each prompt")

# Arrrgs

model_init.add_args(parser)
args = parser.parse_args()

# Prompt templates/modes

if args.modes:
    print(" -- Available formats:")
    for k, v in prompt_formats.items():
        print(f" --   {k:12} : {v().description}")
    sys.exit()

username = args.username
botname = args.botname
system_prompt = args.system_prompt

mode = 'chatml'

prompt_format = prompt_formats[mode]()
prompt_format.botname = botname
prompt_format.username = username
if system_prompt is None: system_prompt = prompt_format.default_system_prompt()

# Initialize model and tokenizer

args.model_dir = "modeldir"

model_init.check_args(args)
model_init.print_options(args)
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(args)
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
model, tokenizer = model_init.init(args, allow_auto_split = True)

# Initialize draft model if provided, assume it always fits on first device

draft_model = None
draft_cache = None

if args.draft_model_dir:

    print(f" -- Draft model: {args.draft_model_dir}")

    draft_config = ExLlamaV2Config()
    draft_config.model_dir = args.draft_model_dir
    draft_config.prepare()

    if draft_config.max_seq_len < model.config.max_seq_len:

        if args.no_draft_scale:
            print(f" !! Warning: Draft model native max sequence length is less than sequence length for model. Speed may decrease after {draft_config.max_seq_len} tokens.")
        else:
            ratio = model.config.max_seq_len / draft_config.max_seq_len
            alpha = -0.13436 + 0.80541 * ratio + 0.28833 * ratio ** 2
            draft_config.scale_alpha_value = alpha
            print(f" -- Applying draft model RoPE alpha = {alpha:.4f}")

    draft_config.max_seq_len = model.config.max_seq_len
    draft_config.no_flash_attn = args.no_flash_attn

    print(" -- Loading draft model...")

    draft_model = ExLlamaV2(draft_config)
    draft_model.load()

    #if args.cache_8bit:
    draft_cache = ExLlamaV2Cache_8bit(draft_model)
    #else:
    #draft_cache = ExLlamaV2Cache(draft_model)

# Create cache

#if args.cache_8bit:
cache = ExLlamaV2Cache_8bit(model, lazy = not model.loaded)
#else:
    #cache = ExLlamaV2Cache(model, lazy = not model.loaded)

# Load model now if auto split enabled

if not model.loaded:

    print(" -- Loading model...")
    model.load_autosplit(cache)

# Chat context

def format_prompt(user_prompt, first):
    global system_prompt, prompt_format

    if first:
        return prompt_format.first_prompt() \
            .replace("<|system_prompt|>", system_prompt) \
            .replace("<|user_prompt|>", user_prompt)
    else:
        return prompt_format.subs_prompt() \
            .replace("<|user_prompt|>", user_prompt)

def encode_prompt(text):
    global tokenizer, prompt_format

    add_bos, add_eos, encode_special_tokens = prompt_format.encoding_options()
    return tokenizer.encode(text, add_bos = add_bos, add_eos = add_eos, encode_special_tokens = encode_special_tokens)

user_prompts = []
responses_ids = []

def get_tokenized_context(max_len):
    global user_prompts, responses_ids

    while True:

        context = torch.empty((1, 0), dtype=torch.long)

        for turn in range(len(user_prompts)):

            up_text = format_prompt(user_prompts[turn], context.shape[-1] == 0)
            up_ids = encode_prompt(up_text)
            context = torch.cat([context, up_ids], dim=-1)

            if turn < len(responses_ids):
                context = torch.cat([context, responses_ids[turn]], dim=-1)

        if context.shape[-1] < max_len: return context

        # If the context is too long, remove the first Q/A pair and try again. The system prompt will be moved to
        # the first entry in the truncated context

        user_prompts = user_prompts[1:]
        responses_ids = responses_ids[1:]


# Generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer, draft_model, draft_cache)

settings = ExLlamaV2Sampler.Settings()
settings.temperature = args.temperature
settings.top_k = args.top_k
settings.top_p = args.top_p
settings.typical = args.typical
settings.token_repetition_penalty = args.repetition_penalty

max_response_tokens = args.max_response_tokens
min_space_in_context = args.response_chunk

# Stop conditions

generator.set_stop_conditions(prompt_format.stop_conditions(tokenizer))

# ANSI color codes

col_default = "\u001b[0m"
col_user = "\u001b[33;1m"  # Yellow
col_bot = "\u001b[34;1m"  # Blue
col_error = "\u001b[31;1m"  # Magenta
col_sysprompt = "\u001b[37;1m"  # Grey

# Code block formatting

codeblock_formatter = None if args.no_code_formatting else CodeBlockFormatter()
in_code_block = False

delim_overflow = ""

# Other options

print_timings = args.print_timings

# Main loop

print(f" -- Prompt format: {args.mode}")
print(f" -- System prompt:")
print()
print(col_sysprompt + system_prompt.strip() + col_default)


from pydantic import BaseModel

class GenerateResponseModel(BaseModel):
    user_message: str
    username: str
    botname: str
    print_timings: Optional[bool] = False
    in_code_block: Optional[bool] = False

from fastapi.responses import StreamingResponse
from typing import Generator
import re

def get_first_prompt(text):
    match = re.search(r'\[image\](.*?)\[/image\]', text)
    return match.group(1) if match else None



def generate_prompt_response(user_message,username,botname,print_timings,in_code_block):
    col_default = "\u001b[0m"
    global responses_ids
    global user_prompts
    up = user_message
    response_output = ""
    responses_ids2 = responses_ids
    user_prompts2 = user_prompts
    user_prompts = []
    responses_ids = []
    user_prompts.append(up)
    # Send tokenized context to generator
    active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)
    generator.begin_stream(active_context, settings)
    
    if prompt_format.print_bot_name():
        response_output += col_bot + botname + ": " + col_default
    
    if print_timings:
        time_begin_stream = time.time()
        if draft_model is not None:
            generator.reset_sd_stats()
    
    while True:
        chunk, eos, tokens = generator.stream()
        if eos:
            responses_ids = responses_ids2
            user_prompts = user_prompts2
            return response_output
        if len(response_output) == 0:
            chunk = chunk.lstrip()
        response_output += chunk
        continue



import re
@app.post("/generate_response/")
async def generate_response(data: GenerateResponseModel) -> StreamingResponse:
    def stream_output():
        global user_prompts
        global responses_ids
        user_message = data.user_message
        username = data.username
        botname = data.botname
        print_timings = data.print_timings
        in_code_block = data.in_code_block
        col_default = "\u001b[0m"

        if (user_message == "clear"):
            user_prompts = []
            responses_ids = []
            return
        pattern = r"inject_response=Q:(.*?)A:(.*)"
        match = re.match(pattern, user_message)

        if (user_message == "history"):
            ite = 0
            for response_tokens in responses_ids:
                response_text = tokenizer.decode(response_tokens[0])
                yield "USER: " + user_prompts[ite]
                yield "\n"
                yield "------------------"
                yield "\n"
                yield response_text
                yield "\n"
                ite = ite + 1
            return
        if match:
            question, response = match.groups()
            user_prompts = [question]
            response = tokenizer.encode(response)
            response_tokens = torch.tensor(response)
            responses_ids = [response_tokens]
            return

        phrases = ["generate an image", "create an image", "make an image", "want an image", "give me an image", "provide an image", "show me an image","generate me an image","generate a photo","create a photo","make a photo","generate a photo"]
        if any(phrase in user_message.lower() for phrase in phrases):
            final_message = '''
            I want you to parse an user message and put [image][/image] tags around image requests.

            Example input:
            "generate the image of a bear, my son wants to see it!"
            Output:
            [image]a bear[/image]    
            Example input:
            "generate an image about oppeheimer, and write a 1000 word essay about him"
            Output:
            [image]oppeheimer[image]
            Example input:
            "I want an image of a beautiful woman in a red dress!"
            Output:
            [image]a beautiful woman in a red dress[/image]
            Example input:
            "Generate a photo of Obama."
            Output:
            [image]photography of Barack Obama[/image]


            Always using [image][/image] tags surrounding the parsed request!

            
            I need you to identify which image the user asked, and put [image][/image] prompts around it.  
            Only provide a reply with the tags. Do not output anything else.     

            The user asked this, identify the image and surround it with tags: '''"\""+user_message+"\""'''

            '''
            prompt = generate_prompt_response(final_message,username,botname,print_timings,in_code_block)
            prompt = get_first_prompt(prompt)
            try:
                stable_diffusion("best quality, 8k uhd,"+prompt)
            except:
                yield '''Sorry, there was a problem when I was trying to generate the image.'''
            # Read the 'generated_image.png' file that your code produced
            with open("generated_image.png", "rb") as image_file:
                # Convert the image to Base64
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            # Embed the Base64 string into the <img> tag
            if prompt:
                img_tag = f'''Generated an image with the prompt "{prompt}" : \n <img src="data:image/png;base64,{encoded_string}" alt="Embedded Image" />'''+"\n"
                yield img_tag
        up = user_message
        
        user_prompts.append(up)

        active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)
        generator.begin_stream(active_context, settings)

        if prompt_format.print_bot_name():
            yield col_bot + botname + ": " + col_default

        response_tokens = 0
        response_text = ""
        responses_ids.append(torch.empty((1, 0), dtype = torch.long))

        if print_timings:
            time_begin_stream = time.time()
            if draft_model is not None: 
                generator.reset_sd_stats()

        while True:
            chunk, eos, tokens = generator.stream()
            yield chunk
            if len(response_text) == 0: chunk = chunk.lstrip()
            response_text += chunk
            responses_ids[-1] = torch.cat([responses_ids[-1], tokens], dim=-1)

            chunk, codeblock_delimiter = (chunk, False) if codeblock_formatter is None else codeblock_formatter.process_delimiter(chunk)

            '''if not in_code_block and codeblock_delimiter:
                codeblock_formatter.begin()
                yield "\n"
                in_code_block = True
                codeblock_delimiter = False

            if in_code_block:
                yield codeblock_formatter.get_code_block(chunk)
            else:
                yield chunk

            if in_code_block and codeblock_delimiter:
                if eos: yield codeblock_formatter.get_code_block("\n")
                yield "\033[0m"
                in_code_block = False
                codeblock_delimiter = False
                '''

            if generator.full():
                active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)
                generator.begin_stream(active_context, settings)

            response_tokens += 1
            if response_tokens == max_response_tokens:
                if tokenizer.eos_token_id in generator.stop_tokens:
                    responses_ids[-1] = torch.cat([responses_ids[-1], tokenizer.single_token(tokenizer.eos_token_id)], dim=-1)
                yield "\n"
                yield col_error + f" !! Response exceeded {max_response_tokens} tokens and was cut short." + col_default
                break

            if eos:
                if prompt_format.print_extra_newline():
                    yield "\n"
                break

        if print_timings:
            time_end_stream = time.time()
            speed = response_tokens / (time_end_stream - time_begin_stream)
            if draft_model is not None:
                eff, acc, _, _, _ = generator.get_sd_stats()
                sd_stats = f", SD eff. {eff*100:.2f}%, SD acc. {acc*100:.2f}%"
            else:
                sd_stats = ""
            yield "\n"
            yield col_sysprompt + f"(Response: {response_tokens} tokens, {speed:.2f} tokens/second{sd_stats})" + col_default

    return StreamingResponse(stream_output())

# If running the app directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
