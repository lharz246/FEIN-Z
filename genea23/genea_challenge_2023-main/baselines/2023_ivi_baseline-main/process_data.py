import numpy as np
import h5py
import librosa
import os
import io
import string
from tqdm import tqdm
from sklearn.pipeline import Pipeline
import joblib as jl
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *
import librosa
import librosa.display
import soundfile as sf
import argparse

from tool import *


def load_metadata(metadata, participant):
    assert participant in ("main-agent", "interloctr"), "`participant` must be either 'main-agent' or 'interloctr'"

    metadict_byfname = {}
    metadict_byindex = {}
    speaker_ids = []
    finger_info = []
    with open(metadata, "r") as f:
        # NOTE: The first line contains the csv header so we skip it
        for i, line in enumerate(f.readlines()[1:]):
            (
                fname,
                main_speaker_id,
                main_has_finger,
                ilocutor_speaker_id,
                ilocutor_has_finger,
            ) = line.strip().split(",")

            if participant == "main-agent":
                has_finger = (main_has_finger == "finger_incl")
                speaker_id = int(main_speaker_id) - 1
            else:
                has_finger = (ilocutor_has_finger == "finger_incl")
                speaker_id = int(ilocutor_speaker_id) - 1

            finger_info.append(has_finger)
            speaker_ids.append(speaker_id)

            metadict_byindex[i] = has_finger, speaker_id
            metadict_byfname[fname + f"_{participant}"] = has_finger, speaker_id

    speaker_ids = np.array(speaker_ids)
    finger_info = np.array(finger_info)
    num_speakers = np.unique(speaker_ids).shape[0]
    # assert num_speakers == spks.max(), "Error speaker info!"
    # print("Number of speakers: ", num_speakers)
    # print("Has Finger Ratio:", np.mean(finger_info))

    return num_speakers, metadict_byfname, metadict_byindex


def load_bvh_jointselector(bvhfile):
    parser = BVHParser()
    parsed_data = parser.parse(bvhfile)

    joint = ['b_root', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_neck0', 'b_head',
             'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg',
             'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1',
             'b_l_pinky2', 'b_l_pinky3',
             'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1',
             'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1',
             'b_l_thumb2', 'b_l_thumb3',
             'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0',
             'b_r_thumb1', 'b_r_thumb2',
             'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3',
             'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1',
             'b_r_index2', 'b_r_index3']

    mexp_full = Pipeline([
        ('jtsel', JointSelector(joint, include_root=True)),
        ('param', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover_withroot()),
        ('np', Numpyfier()),
    ])
    fullexpdata = mexp_full.fit_transform([parsed_data])[0]

    mexp_upperbody = Pipeline([
        # ('jtsel', JointSelector(joint, include_root=False)),
        ('param', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover_()),
        ('np', Numpyfier()),
    ])
    upperexpdata = mexp_upperbody.fit_transform([parsed_data])[0]

    return fullexpdata, upperexpdata


def load_audio(audiofile):
    audio, sr = librosa.load(audiofile, sr=None)
    return audio, sr


def load_wordvectors(fname):
    print("Loading word2vector ...")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(v) for v in tokens[1:]])
    return data


def load_tsv(tsvfile):
    sentences = []
    sentence = []
    offset = 0
    with open(tsvfile, "r") as f:
        for i, line in enumerate(f.readlines()):
            start, end, raw_word = line.strip().split("\t")
            start = float(start)
            end = float(end)

            if start - offset > .05 and i > 0:
                if sentence[-1][1] - sentence[0][0] > .2:  # if duration is long enough
                    sentences.append(sentence)
                sentence = [[start, end, raw_word]]
            else:
                sentence.append([start, end, raw_word])

            offset = end

    durations = [s[-1][1] - s[0][0] for s in sentences]
    sentence_lengths = [len(s) for s in sentences]
    return sentences, durations, sentence_lengths


def load_tsv_unclipped(tsvfile):
    sentence = []
    with open(tsvfile, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split("\t")
            if len(line) == 3:
                start, end, raw_word = line
                start = float(start)
                end = float(end)
                sentence.append([start, end, raw_word])
    return sentence


def find_timestamp_from_timings(timestamp, timings):
    output = None
    for i, (start, end) in enumerate(timings):
        if start <= timestamp < end:
            output = i
            break
    return output


def prepare_h5_unclipped(dataroot, h5file, participant, word2vector,do_gesture=True):
    assert participant in ("main-agent", "interloctr"), "`participant` must be either 'main-agent' or 'interloctr'"

    metadata_path = os.path.join(dataroot, "metadata.csv")
    num_speakers, metadict_byfname, metadict_byindex = load_metadata(metadata_path, participant)
    filenames = sorted(metadict_byfname.keys())

    wavdir = os.path.join(dataroot, participant, "wav")
    tsvdir = os.path.join(dataroot, participant, "tsv")
    bvhdir = os.path.join(dataroot, participant, "bvh")

    with h5py.File(h5file, "w") as h5:
        for i, filename in enumerate(filenames):
            print("{}/{} {}              ".format(i + 1, len(filenames), filename), end="\r")
            g_data = h5.create_group(str(i))

            hasfinger, speaker_id = metadict_byfname[filename]
            audio, sr = load_audio(os.path.join(wavdir, filename + ".wav"))
            prosody = extract_prosodic_features(os.path.join(wavdir, filename + ".wav"))
            mfcc = calculate_mfcc(audio, sr)
            melspec = calculate_spectrogram(audio, sr)
            
            if do_gesture:
                full, upper = load_bvh_jointselector(os.path.join(bvhdir, filename + ".bvh"))

            if do_gesture:
                crop_length = min(mfcc.shape[0], prosody.shape[0], melspec.shape[0], full.shape[0], upper.shape[0])
            else:
                crop_length = min(mfcc.shape[0], prosody.shape[0], melspec.shape[0])
            prosody = prosody[:crop_length]
            mfcc = mfcc[:crop_length]
            melspec = melspec[:crop_length]
            if do_gesture:
                full = full[:crop_length]
                upper = upper[:crop_length]

            g_audiodata = g_data.create_group("audio")
            g_motiondata = g_data.create_group("motion")
            g_data.create_dataset("has_finger", data=[hasfinger])
            g_data.create_dataset("speaker_id", data=[speaker_id])

            # g_audiodata.create_dataset("raw_audio", data=(audio*32768).astype(np.int16), dtype=np.int16)
            g_audiodata.create_dataset("mfcc", data=mfcc, dtype=np.float32)
            g_audiodata.create_dataset("melspectrogram", data=melspec, dtype=np.float32)
            g_audiodata.create_dataset("prosody", data=prosody, dtype=np.float32)
            if do_gesture:
                g_motiondata.create_dataset("expmap_full", data=full, dtype=np.float32)
                g_motiondata.create_dataset("expmap_upper", data=upper, dtype=np.float32)

            # Process the txt
            # Align txt with audio
            textfeatures = np.zeros([crop_length, 300 + 2])
            textfeatures[:, -1] = 1
            sentence = load_tsv_unclipped(os.path.join(tsvdir, filename + ".tsv"))

            for wi, (start, end, raw_word) in enumerate(sentence):
                has_laughter = "#" in raw_word
                start_frame = int(start * 30)
                end_frame = int(end * 30)
                textfeatures[start_frame:end_frame, -1] = 0

                word = raw_word.translate(str.maketrans('', '', string.punctuation))
                word = word.strip()
                word = word.replace("  ", " ")

                if len(word) > 0:
                    if word[0] == " ":
                        word = word[1:]

                if " " in word:
                    ww = word.split(" ")
                    subword_duration = (end_frame - start_frame) / len(ww)
                    for j, w in enumerate(ww):
                        vector = word2vector.get(w)
                        if vector is not None:
                            ss = start_frame + int(subword_duration * j)
                            ee = start_frame + int(subword_duration * (j + 1))
                            textfeatures[ss:ee, :300] = vector
                else:
                    vector = word2vector.get(word)
                    if vector is not None:
                        textfeatures[start_frame:end_frame, :300] = vector
                textfeatures[start_frame:end_frame, -2] = has_laughter
            g_data.create_dataset("text", data=textfeatures, dtype=np.float32)
    print()
    return


def prepare_h5_unclipped_test(metadata, h5file="tst_v0.h5"):
    num_speakers, metadict_byfname, metadict_byindex = load_metadata(metadata)
    word2vector = load_wordvectors(fname="crawl-300d-2M.vec")
    filenames = sorted(metadict_byfname.keys())

    with h5py.File(h5file, "w") as h5:
        for i, filename in enumerate(filenames):
            print("{}/{} {}              ".format(i + 1, len(filenames), filename), end="\r")
            g_data = h5.create_group(str(i))
            hasfinger, speaker_id = metadict_byfname[filename]
            audio, sr = load_audio(os.path.join(wavdir, filename + ".wav"))
            prosody = extract_prosodic_features(os.path.join(wavdir, filename + ".wav"))
            mfcc = calculate_mfcc(audio, sr)
            melspec = calculate_spectrogram(audio, sr)

            crop_length = min(mfcc.shape[0], prosody.shape[0], melspec.shape[0])
            prosody = prosody[:crop_length]
            mfcc = mfcc[:crop_length]
            melspec = melspec[:crop_length]

            g_audiodata = g_data.create_group("audio")
            g_data.create_dataset("has_finger", data=[hasfinger])
            g_data.create_dataset("speaker_id", data=[speaker_id])

            # g_audiodata.create_dataset("raw_audio", data=(audio * 32768).astype(np.int16), dtype=np.int16)
            g_audiodata.create_dataset("mfcc", data=mfcc, dtype=np.float32)
            g_audiodata.create_dataset("melspectrogram", data=melspec, dtype=np.float32)
            g_audiodata.create_dataset("prosody", data=prosody, dtype=np.float32)

            # Process the txt
            # Align txt with audio
            textfeatures = np.zeros([crop_length, 300 + 2])
            textfeatures[:, -1] = 1
            sentence = load_tsv_unclipped(os.path.join(tsvdir, filename + ".tsv"))

            for wi, (start, end, raw_word) in enumerate(sentence):
                has_laughter = "#" in raw_word
                start_frame = int(start * 30)
                end_frame = int(end * 30)
                textfeatures[start_frame:end_frame, -1] = 0

                word = raw_word.translate(str.maketrans('', '', string.punctuation))
                word = word.strip()
                word = word.replace("  ", " ")

                if len(word) > 0:
                    if word[0] == " ":
                        word = word[1:]

                if " " in word:
                    ww = word.split(" ")
                    subword_duration = (end_frame - start_frame) / len(ww)
                    for j, w in enumerate(ww):
                        vector = word2vector.get(w)
                        if vector is not None:
                            ss = start_frame + int(subword_duration * j)
                            ee = start_frame + int(subword_duration * (j + 1))
                            textfeatures[ss:ee, :300] = vector
                else:
                    vector = word2vector.get(word)
                    if vector is not None:
                        textfeatures[start_frame:end_frame, :300] = vector
                textfeatures[start_frame:end_frame, -2] = has_laughter
            g_data.create_dataset("text", data=textfeatures, dtype=np.float32)
    print()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=str, default="/mnt/ssd-tb/Genea/genea2023_trn/genea2023_dataset/")
    args = parser.parse_args()
    print(os.getcwd())
    word2vector = load_wordvectors(
        fname="/mnt/techfak_compute/genea23/genea_challenge_2023-main/baselines/2023_ivi_baseline-main/crawl-300d-2M.vec")

    dataset_type = "trn"
    dataroot = os.path.join(args.dataset_path, dataset_type)
    wavdir = os.path.join(dataroot, "wav")
    tsvdir = os.path.join(dataroot, "tsv")
    bvhdir = os.path.join(dataroot, "bvh")

    #prepare_h5_unclipped(dataroot, f"{dataset_type}_main-agent_v0.h5", "main-agent", word2vector)
    #prepare_h5_unclipped(dataroot, f"{dataset_type}_interloctr_v0.h5", "interloctr", word2vector)

    dataset_type = "val"
    dataroot = os.path.join(args.dataset_path, dataset_type)

    #prepare_h5_unclipped(dataroot, f"{dataset_type}_main-agent_v0.h5", "main-agent", word2vector)
    #prepare_h5_unclipped(dataroot, f"{dataset_type}_interloctr_v0.h5", "interloctr", word2vector)

    dataset_type = "tst"
    dataroot = os.path.join(args.dataset_path, dataset_type)
    
    prepare_h5_unclipped(dataroot, f"{dataset_type}_main-agent_v0.h5", "main-agent", word2vector,do_gesture=False)
    prepare_h5_unclipped(dataroot, f"{dataset_type}_interloctr_v0.h5", "interloctr", word2vector)
