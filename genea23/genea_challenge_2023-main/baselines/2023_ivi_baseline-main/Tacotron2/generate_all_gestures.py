import argparse
import librosa
from time import time
import os
import time
import math
import h5py
import numpy as np
from tqdm import tqdm
import joblib as jl
import torch
from pymo.viz_tools import *
from pymo.writers import *
from common.test_model import *
from common.hparams_bcn2 import create_hparams
from common.loss_bcgg import BCGLoss
from scipy.signal import savgol_filter


parser = argparse.ArgumentParser()
parser.add_argument('-f', "--h5file", type=str, default="/mnt/techfak_compute/genea23/genea_challenge_2023-main/baselines/2023_ivi_baseline-main/val_main-agent_v0.h5")
parser.add_argument('-ch', "--checkpoint_path", type=str, required=True)
parser.add_argument('-o', "--output_dir", type=str, default="outputs")
parser.add_argument('-t', "--track", type=str, default="full", help="The track for the bvh files. Can only be either 'full' or 'upper'")
# parser.add_argument('-p', "--pipeline", default="../pipeline_expmap_full.sav", type=str)
# parser.add_argument('-p', "--pipeline", default="../pipeline_expmap_upper.sav", type=str)
args = parser.parse_args()

print("Predicting {} body motion.".format(args.track))

hparams = create_hparams()
torch.cuda.set_device("cuda:{}".format(hparams.device))
torch.backends.cudnn.enabled = hparams.cudnn_enabled
torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
torch.manual_seed(hparams.seed)
torch.cuda.manual_seed(hparams.seed)
configname = args.checkpoint_path.split("/")[0]
args.output_dir = os.path.join(args.output_dir, configname)
os.makedirs(args.output_dir, exist_ok=True)

if args.track == "full":
	hparams.n_acoustic_feat_dims = 168
else:
	hparams.n_acoustic_feat_dims = 57


### Load Tacotron2 Model
model = BCNetwork(hparams)
if args.checkpoint_path is not None:
	model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu")['state_dict'])
model.cuda().eval()

# Load postprocessing pipeline
npy_root = "/mnt/techfak_compute/genea23/genea_challenge_2023-main/baselines/2023_ivi_baseline-main/"
mel_mean = np.load(os.path.join(npy_root, "mel_mean.npy"))
mel_std = np.load(os.path.join(npy_root, "mel_std.npy"))
mfcc_mean = np.load(os.path.join(npy_root, "mfcc_mean.npy"))
mfcc_std = np.load(os.path.join(npy_root, "mfcc_std.npy"))
prosody_mean = np.load(os.path.join(npy_root, "prosody_mean.npy"))
prosody_std = np.load(os.path.join(npy_root, "prosody_std.npy"))
h5 = h5py.File(args.h5file, "r")

print(args.output_dir)
for index in tqdm(range(5)): #len(h5.keys())
	### Load input
	mel = torch.FloatTensor((h5[str(index)]["audio"]["melspectrogram"][:] - mel_mean) / mel_std)
	mfcc = torch.FloatTensor((h5[str(index)]["audio"]["mfcc"][:] - mfcc_mean) / mfcc_std)
	prosody = torch.FloatTensor((h5[str(index)]["audio"]["prosody"][:] - prosody_mean) / prosody_std)
	text = torch.FloatTensor(h5[str(index)]["text"][:])
	motion = torch.FloatTensor(h5[str(index)]["motion"]["expmap_full"][:])
	# print(wu_gesture.shape)
	wu_ges = motion[:200,:]

	speaker = torch.zeros([mel.shape[0], 17])
	speaker[:, h5[str(index)]["speaker_id"][:]] = 1

	audio = np.concatenate((mel, mfcc, prosody), axis=-1)
	text = np.concatenate((text, speaker), axis=-1)
	audio = torch.FloatTensor(audio).T
	text = torch.FloatTensor(text).T
	wu_ges = wu_ges.T
	# motion = torch.unsqueeze(motion, 0)
	# print(f'audio shape real: {audio.shape}')
	# print(f'text shape real: {text.shape}')
	test_loss = BCGLoss({'joints': 1, 'velocity': 5, 'acceleration': 10})


	### Inference
	with torch.no_grad():
		motion = torch.unsqueeze(motion, 0)
		y_pred = model.inference(audio, text, wu_ges)
		predicted_gesture = y_pred
		predicted_gesture = predicted_gesture.transpose(1,2)
		# print(text.shape)
		# print(predicted_gesture.shape)
		# print(motion.shape)
		print(f'test loss: {test_loss(predicted_gesture, motion.transpose(1,2).cuda())}')
		predicted_gesture = predicted_gesture.squeeze(0).cpu().detach().numpy()

	# todo: convert to bvh and save to output folder
	predicted_gesture = savgol_filter(predicted_gesture, 9, 3, axis=0)
	# print(predicted_gesture.shape)
	# exit()
	
	if args.track == "full":
		#predicted_gesture[:, 21:24] = np.mean(predicted_gesture[:, 21:24], axis=0)
		#predicted_gesture[:, 27] = np.clip(predicted_gesture[:, 27], -9999, 0.6)
		#predicted_gesture[:, 39] = np.clip(predicted_gesture[:, 39], -9999, -2.0)
		#predicted_gesture[:, 40] = np.clip(predicted_gesture[:, 40], -9999, 0.4)
		#predicted_gesture[:, 41] = np.clip(predicted_gesture[:, 41], -9999, 0.4)
		#predicted_gesture[:, 45] = np.clip(predicted_gesture[:, 45], -9999, 0.6)
		#predicted_gesture[:, 12] = np.clip(predicted_gesture[:, 12], -9999, 1.4)
		#predicted_gesture[:, 3] = np.clip(predicted_gesture[:, 3], -9999, 1.4)
		pipeline = jl.load("/mnt/techfak_compute/genea23/genea_challenge_2023-main/baselines/2023_ivi_baseline-main/pipeline_expmap_full.sav")

	else:
		#predicted_gesture[:, 21-18:24-18] = np.mean(predicted_gesture[:, 21-18:24-18], axis=0)
		#predicted_gesture[:, 27-18] = np.clip(predicted_gesture[:, 27-18], -9999, 0.6)
		#predicted_gesture[:, 39-18:42-18] = np.mean(predicted_gesture[:, 39-18:42-18], axis=0)
		#predicted_gesture[:, 45-18] = np.clip(predicted_gesture[:, 45-18], -9999, 0.6)
		pipeline = jl.load("../pipeline_expmap_upper.sav")

	bvh_data = pipeline.inverse_transform([predicted_gesture])[0]
	writer = BVHWriter()
	with open(os.path.join(args.output_dir, "{}-{:03d}.bvh".format(configname, index)), 'w') as f:
		writer.write(bvh_data, f, framerate=30)

h5.close()

if __name__ == "__main__":
	pass
