import vosk
import soundfile as sf


file = input("Path to .wav file")
audio, samplerate = sf.read(file)

model = vosk.Model(input("Language of the model used (e. g. ru, en, fr, de"))

chunk_size = 4000
for start in range(0, len(audio), chunk_size):
    end = start + chunk_size
    chunk = audio[start:end]

    wave_data = vosk.WaveData(chunk)

    result = model.recognize(wave_data)

    if result.result:
            for hypothesis in result.result:
                if hypothesis.confidence > 0.8:
                    speaker = hypothesis.speaker
                    text = hypothesis.text

                    speaker_dir = os.path.join(output_dir, f"speaker{speaker}")
                    if not os.path.exists(speaker_dir):
                        os.makedirs(speaker_dir)

                    audio_filename = f"speaker{speaker}_{hypothesis.start_time}.wav"
                    audio_data = hypothesis.frames * samplerate
                    sf.write(os.path.join(speaker_dir, audio_filename), audio_data, samplerate)

                    text_filename = f"speaker{speaker}_{hypothesis.start_time}.txt"
                    with open(os.path.join(speaker_dir, text_filename), 'w', encoding='utf-8') as f:
                        f.write(text + '\n')

if __name__ == '__main__':
    audio_file = "dialogue.wav"
    output_dir = "separated_voices"  

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    separate_voices(audio_file, output_dir)
