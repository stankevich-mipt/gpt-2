import librosa as lr
import numpy as np


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x

def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s
    

def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized


def create_chunks(location):
	print("create dataset from audio files at", location)
	files = list_all_audio_files(location)

	processed_files = []

	for i, file in enumerate(files):
    	print("  processed " + str(i) + " of " + str(len(files)) + " files")
    	file_data, _ = lr.load(path=file, 
    						   sr=None,
    						   mono=True)

    	quantized_data = quantize_data(file_data, 256).astype(np.uint8)
        processed_files.append(quantized_data)

    return processed_files

