import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import subprocess
import tempfile
import soundfile as sf


ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)


# reference_speaker = 'resources/example_reference.mp3' # This is the voice you want to clone
reference_speaker = 'resources/my_voice_50s.wav' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)


from melo.api import TTS

texts = {
    'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
    'EN': "Did you ever hear a folk tale about a giant turtle?",
    'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
    'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
    'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
    'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
    # 'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",  # Disabled due to MeCab configuration issues
}


src_path = f'{output_dir}/tmp.wav'

# Speed is adjustable
speed = 1.0

# Options
SAVE_WAV = False
PLAY_AUDIO = True


# language, text = 'EN', 'Wah-way is a pretty good company. Did you ever hear that? It is so true that Wah-way is a pretty good company.'
language, text = 'EN', 'Wah-way!'
model = TTS(language=language, device=device)
speaker_ids = model.hps.data.spk2id

print(speaker_ids.keys()) ## dict_keys(['EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU', 'EN-Default'])
speaker_key = 'EN-US'


speaker_id = speaker_ids[speaker_key]
speaker_key = speaker_key.lower().replace('_', '-')


source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
if torch.backends.mps.is_available() and device == 'cpu':
    torch.backends.mps.is_available = lambda: False



import time
time_start = time.time()

model.tts_to_file(text, speaker_id, src_path, speed=speed)

time_middle = time.time()
print(f'TTS time: {time_middle - time_start}')


# Run the tone color converter
encode_message = "@MyShell"
converted_audio = tone_color_converter.convert(
    audio_src_path=src_path, 
    src_se=source_se, 
    tgt_se=target_se, 
    output_path=None,
    message=encode_message)

# Print basic info about the returned audio array
print(f"Converted audio returned: samples={len(converted_audio)}, dtype={converted_audio.dtype}, sr={tone_color_converter.hps.data.sampling_rate}")

# Optionally save WAV and/or play audio
sample_rate = tone_color_converter.hps.data.sampling_rate
save_path = f'{output_dir}/output_v6_{speaker_key}.wav'
if SAVE_WAV:
    sf.write(save_path, converted_audio, sample_rate)
    print(f"Saved converted audio to: {save_path}")


# Play with a temporary WAV file (no persistent save)
with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpf:
    temp_wav_path = tmpf.name
sf.write(temp_wav_path, converted_audio, sample_rate)
subprocess.run(['afplay', temp_wav_path], check=False)
try:
    os.remove(temp_wav_path)
except Exception:
    pass


time_end = time.time()
print(f'Conversion time: {time_end - time_start}')