import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
import numpy as np
from model.generator import Generator
from utils.hparams import HParam, load_hparam_str
from utils.pqmf import PQMF

MAX_WAV_VALUE = 32768.0


def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                        ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=False)

    with torch.no_grad():
        mel = torch.from_numpy(np.load(args.input))
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.cuda()
        if hp.model.out_channels == 1:
            audio = model.inference(mel).cpu().detach().numpy()
        else:
            pqmf = PQMF()
            audio = pqmf.synthesis(model.inference(mel)).view(-1).cpu().detach().numpy()

        out_path = args.input.replace('.npy', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
        write(out_path, hp.audio.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="directory of mel-spectrograms to invert into raw audio. ")
    args = parser.parse_args()

    main(args)
