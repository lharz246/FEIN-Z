import argparse
import h5py
import joblib as jl
import librosa
import math
import numpy as np
import os
import time
import torch
from scipy.signal import savgol_filter
from time import time
from tqdm import tqdm

from common.bcn3 import BCNetwork
from common.hparams_bcn3 import create_hparams
from common.loss_bcn3 import *
from pymo.viz_tools import *
from pymo.writers import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', "--h5file", type=str,
                    default="/mnt/techfak_compute/genea/genea23/genea_challenge_2023-main/baselines/2023_ivi_baseline-main/val_main-agent_v0.h5")
parser.add_argument('-fi', "--h5file_interlocutor", type=str,
                    default="/mnt/techfak_compute/genea/genea23/genea_challenge_2023-main/baselines/2023_ivi_baseline-main/val_interloctr_v0.h5")
parser.add_argument('-ch', "--checkpoint_path", type=str, required=True)
parser.add_argument('-o', "--output_dir", type=str, default="outputs")
parser.add_argument('-t', "--track", type=str, default="full",
                    help="The track for the bvh files. Can only be either 'full' or 'upper'")
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
model = BCNetwork(hparams.model_dyadic)
model.eval()
if args.checkpoint_path is not None:
    print("loading")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu")['state_dict'], strict=True)
model.cuda().eval()

# Load postprocessing pipeline
npy_root = "/mnt/techfak_compute/genea/genea23/genea_challenge_2023-main/baselines/2023_ivi_baseline-main/"
mel_mean = np.load(os.path.join(npy_root, "mel_mean.npy"))
mel_std = np.load(os.path.join(npy_root, "mel_std.npy"))
mfcc_mean = np.load(os.path.join(npy_root, "mfcc_mean.npy"))
mfcc_std = np.load(os.path.join(npy_root, "mfcc_std.npy"))
prosody_mean = np.load(os.path.join(npy_root, "prosody_mean.npy"))
prosody_std = np.load(os.path.join(npy_root, "prosody_std.npy"))
import pickle

mins, maxs, median, std, v1_maxs, a1_maxs, j1_maxs = pickle.load(open("/mnt/techfak_compute/genea/genea23/genea_challenge_2023-main/baselines/2023_ivi_baseline-main/Tacotron2/inf_val.p", "rb"))


def get_vaj(gesture, i):
    v1 = gesture[i] - gesture[i - 1]
    v2 = gesture[i - 1] - gesture[i - 2]
    v3 = gesture[i - 2] - gesture[i - 3]
    a1 = v1 - v2
    a2 = v2 - v3
    j1 = a1 - a2
    return v1, a1, j1


def normalize(mins, maxs, median, std, vg, ag, jg, gesture):
    gesture = torch.from_numpy(gesture)
    # gesture = gesture.double()
    # for i in range(168):
    #    gesture[:,i] = torch.clip(gesture[:,i],mins[i],maxs[i])

    for i in range(4, gesture.shape[0]):
        v1, a1, j1 = get_vaj(gesture, i)

        for j in tqdm(range(v1.shape[0])):
            did = False
            initial = v1[j]
            while v1[j] > vg[j]:
                did = True
                gesture[i, j] = torch.lerp(gesture[i, j], gesture[i - 1, j], 0.1)
                v1, a1, j1 = get_vaj(gesture, i)
            if did:
                print("j_error", j, "-- max is:", vg[j], "value was:", initial, "value now:", v1[j])

        for j in range(v1.shape[0]):
            did = False
            initial = j1[j]
            while a1[j] > ag[j]:
                did = True
                gesture[i - 1, j] = torch.lerp(gesture[i - 1, j], gesture[i - 2, j], 0.1)
                gesture[i, j] = torch.lerp(gesture[i, j], gesture[i - 1, j], 0.1)
                v1, a1, j1 = get_vaj(gesture, i)
            if did:
                print("j_error", j, "-- max is:", ag[j], "value was:", initial, "value now:", a1[j])

        for j in range(v1.shape[0]):
            did = False
            initial = j1[j]
            while j1[j] > jg[j]:
                did = True
                gesture[i - 2, j] = torch.lerp(gesture[i - 2, j], gesture[i - 3, j], 0.1)
                gesture[i - 1, j] = torch.lerp(gesture[i - 1, j], gesture[i - 2, j], 0.1)
                gesture[i, j] = torch.lerp(gesture[i, j], gesture[i - 1, j], 0.1)
                v1, a1, j1 = get_vaj(gesture, i)
            if did:
                print("j_error", j, "-- max is:", jg[j], "value was:", initial, "value now:", j1[j])

        # for i in range(168):
        #    gesture[:, i] = torch.clip(gesture[:, i], mins[i], maxs[i])

    return gesture.numpy()


h5_interlocutor = h5py.File(args.h5file_interlocutor, "r")
h5 = h5py.File(args.h5file, "r")
criterion = BCGLoss({'joints': 10, 'velocity': 5, 'acceleration': 2})
for index in tqdm(range(len(h5.keys()))):  # len(h5.keys())
    ### Main input
    mel = torch.FloatTensor((h5[str(index)]["audio"]["melspectrogram"][:] - mel_mean) / mel_std)
    mfcc = torch.FloatTensor((h5[str(index)]["audio"]["mfcc"][:] - mfcc_mean) / mfcc_std)
    prosody = torch.FloatTensor((h5[str(index)]["audio"]["prosody"][:] - prosody_mean) / prosody_std)
    text = torch.FloatTensor(h5[str(index)]["text"][:])
    speaker = torch.zeros([mel.shape[0], 17])
    speaker[:, h5[str(index)]["speaker_id"][:]] = 1
    # motion = torch.FloatTensor(h5[str(index)]["motion"]["expmap_full"][:])
    # motion = motion.transpose(0, 1).unsqueeze(0).cuda()
    audio = torch.cat((mel, mfcc, prosody), axis=-1)
    audio = audio.transpose(0, 1).unsqueeze(0).cuda()
    text = torch.cat((text, speaker), axis=-1)
    text = text.transpose(0, 1).unsqueeze(0).cuda()
    # print(audio.shape)

    ### Interlocutor input
    mel_interlocutor = torch.FloatTensor(
        (h5_interlocutor[str(index)]["audio"]["melspectrogram"][:] - mel_mean) / mel_std)
    mfcc_interlocutor = torch.FloatTensor((h5_interlocutor[str(index)]["audio"]["mfcc"][:] - mfcc_mean) / mfcc_std)
    prosody_interlocutor = torch.FloatTensor(
        (h5_interlocutor[str(index)]["audio"]["prosody"][:] - prosody_mean) / prosody_std)
    text_interlocutor = torch.FloatTensor(h5_interlocutor[str(index)]["text"][:])
    speaker_interlocutor = torch.zeros([mel_interlocutor.shape[0], 17])
    speaker_interlocutor[:, h5_interlocutor[str(index)]["speaker_id"][:]] = 1
    # gestures = torch.FloatTensor(h5[str(index)]["motion"]["expmap_full"][:])
    # gestures = gestures.transpose(0, 1).unsqueeze(0).cuda()
    # print(gestures.shape)

    inter_gestures = torch.FloatTensor(h5_interlocutor[str(index)]["motion"]["expmap_full"][:])
    inter_gestures = inter_gestures.transpose(0, 1).unsqueeze(0).cuda()
    gestures = torch.zeros((1, 168, inter_gestures.shape[2])).cuda()
    audio_interlocutor = torch.cat((mel_interlocutor, mfcc_interlocutor, prosody_interlocutor), axis=-1)
    text_interlocutor = torch.cat((text_interlocutor, speaker_interlocutor), axis=-1)
    audio_interlocutor = audio_interlocutor.transpose(0, 1).unsqueeze(0).cuda()
    text_interlocutor = text_interlocutor.transpose(0, 1).unsqueeze(0).cuda()

    # print(f'audio shape:{audio.shape}')
    # print(f'text shape:{text.shape}')
    # print(f'motion shape:{motion.shape}')
    ### Concatenating speaker and interlocutor inputs
    seqlen = audio.shape[-1]
    warmup_period = 50
    # audio *= 0
    # text *= 0
    # audio_interlocutor *= 0
    # text_interlocutor *= 0
    input_x = audio, text, audio_interlocutor, text_interlocutor, seqlen, gestures, inter_gestures
    model.eval()
    with torch.no_grad():
        audio = torch.cat([audio, audio_interlocutor], dim=1)
        text = torch.cat([text, text_interlocutor], dim=1)
        gestures_in = torch.cat([gestures, inter_gestures], dim=1)
        x = (audio, text, seqlen,
             torch.FloatTensor([0]), gestures_in)
        gestures_pred = model.inference(x)
        # gestures = gestures[:, :, 50:]
        # loss_bc = criterion(gestures_pred, gestures)

    predicted_gesture = gestures_pred.squeeze(0).transpose(0, 1).cpu().detach().numpy()
    #print(predicted_gesture.shape)
    # todo: convert to bvh and save to output folder
    # predicted_gesture = normalize(mins, maxs, median, std, v1_maxs, a1_maxs, j1_maxs, predicted_gesture)
    #predicted_gesture = savgol_filter(predicted_gesture, 20, 3, axis=0)
    # print(predicted_gesture.shape)
    # exit()

    if args.track == "full":
        # predicted_gesture[:, 21:24] = np.mean(predicted_gesture[:, 21:24], axis=0)
        # predicted_gesture[:, 27] = np.clip(predicted_gesture[:, 27], -9999, 0.6)
        # predicted_gesture[:, 39] = np.clip(predicted_gesture[:, 39], -9999, -2.0)
        # predicted_gesture[:, 40] = np.clip(predicted_gesture[:, 40], -9999, 0.4)
        # predicted_gesture[:, 41] = np.clip(predicted_gesture[:, 41], -9999, 0.4)
        # predicted_gesture[:, 45] = np.clip(predicted_gesture[:, 45], -9999, 0.6)
        # predicted_gesture[:, 12] = np.clip(predicted_gesture[:, 12], -9999, 1.4)
        # predicted_gesture[:, 3] = np.clip(predicted_gesture[:, 3], -9999, 1.4)
        pipeline = jl.load(
            "/mnt/techfak_compute/genea/genea23/genea_challenge_2023-main/baselines/2023_ivi_baseline-main/pipeline_expmap_full.sav")

    else:
        # predicted_gesture[:, 21 - 18:24 - 18] = np.mean(predicted_gesture[:, 21 - 18:24 - 18], axis=0)
        # predicted_gesture[:, 27 - 18] = np.clip(predicted_gesture[:, 27 - 18], -9999, 0.6)
        # predicted_gesture[:, 39 - 18:42 - 18] = np.mean(predicted_gesture[:, 39 - 18:42 - 18], axis=0)
        # predicted_gesture[:, 45 - 18] = np.clip(predicted_gesture[:, 45 - 18], -9999, 0.6)
        pipeline = jl.load("../pipeline_expmap_upper.sav")

    bvh_data = pipeline.inverse_transform([predicted_gesture])[0]
    writer = BVHWriter()

    limiter = {

        "b_l_arm_twist_Xrotation": (0, 45),#eventuell 90
        "b_l_arm_twist_Yrotation": (0, 0),
        "b_l_arm_twist_Zrotation": (0, 0),

        "b_r_arm_twist_Xrotation": (0, 45),
        "b_r_arm_twist_Yrotation": (0, 0),
        "b_r_arm_twist_Zrotation": (0, 0),



        "b_l_wrist_twist_Xrotation": (-90, -10),#eventuell +0
        "b_l_wrist_twist_Yrotation": (-10, 45),
        "b_l_wrist_twist_Zrotation": (-5, 5),

        "b_r_wrist_twist_Xrotation": (-90,-10),
        "b_r_wrist_twist_Yrotation": (-10, 45),
        "b_r_wrist_twist_Zrotation": (-5, 5),

    }
    for k,(low,high) in limiter.items():
        # print(bvh_data.values[k].describe(percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]))
        # print("-----")
        # print()
        bvh_data.values[k].clip(lower=low,upper=high,inplace=True)
    #print("done")

    quants = pickle.load(open("quants.p", "rb"))
    for col in quants.keys():
        if col in list(bvh_data.values.columns.values):
            p2,p5,p95,p98 = quants[col]
            # print(col)
            bvh_data.values[col].clip(lower=p2, upper=p98, inplace=True)

            bvh_data.values[col] = bvh_data.values[col].rolling(window=12,min_periods=1).mean()

    #bvh_data = pipeline.transform([bvh_data])[0]
    #bvh_data = savgol_filter(bvh_data, 20, 3, axis=0)
    #bvh_data = pipeline.inverse_transform([predicted_gesture])[0]

    # print(bvh_data.values.fillna(0)[bvh_data.values > 360].to_string())

    with open(os.path.join(args.output_dir, "tst_2023_v0_{:03d}_main-agent.bvh".format(index)), 'w') as f:
        writer.write(bvh_data, f, framerate=30)

h5_interlocutor.close()
h5.close()

if __name__ == "__main__":
    pass
