import whisperx
import gc 

device = "cuda" 
audio_file = "audio.m4a"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)


def segments_to_srt(segments):
    srt_content = ""
    
    for idx, segment in enumerate(segments):
        # Start and end times
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        
        # Speaker and text
        try:
            speaker = segment['speaker']
        except:
            speaker = 'unknown'
        text = f"{speaker}: {segment['text']}"
        
        # Append to the SRT content
        srt_content += f"{idx + 1}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"{text}\n\n"
    
    return srt_content

def format_time(seconds):
    """Convert seconds to SRT time format."""
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    ms = (s - int(s)) * 1000
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int(ms):03}"

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token='HF_TOKEN', device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs

# Use the function to convert segments to SRT
srt_output = segments_to_srt(result["segments"])

# Save to an SRT file
with open("output.srt", "w", encoding="utf-8") as file:
    file.write(srt_output)
