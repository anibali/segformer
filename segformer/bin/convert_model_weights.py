"""Convert model weight files provided by the SegFormer paper authors.

This script takes official weight files (available on Google Drive [1, 2]) and converts them into
PyTorch Hub-friendly versions which we host on GitHub [3].

[1]: https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA
[2]: https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia
[3]: https://github.com/anibali/segformer/releases/tag/v0.0.0
"""

import argparse
import os.path
import re
import sys
import hashlib

import torch


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='path to input model file')
    return parser


def main(args):
    opts = argument_parser().parse_args(args)
    data = torch.load(opts.input, map_location='cpu')

    pattern = re.compile(r'mit_(b\d).pth')
    match = pattern.fullmatch(os.path.basename(opts.input))
    tmp_file = 'tmp.pth'

    if match:
        stem = f'segformer_{match[1]}_backbone_imagenet'
        state_dict = data
        # Remove parameters that are not actually used by the model.
        del state_dict['head.weight']
        del state_dict['head.bias']
    else:
        stem = os.path.splitext(os.path.basename(opts.input))[0].replace('.', '_')
        state_dict = data['state_dict']
        # Remove parameters that are not actually used by the model.
        del state_dict['decode_head.conv_seg.weight']
        del state_dict['decode_head.conv_seg.bias']

    torch.save(state_dict, tmp_file)
    sha256 = hashlib.sha256()
    with open(tmp_file, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    checksum = sha256.hexdigest()[:8]
    out_path = os.path.abspath(f'{stem}-{checksum}.pth')
    os.rename(tmp_file, out_path)
    print(f'Output written to {out_path}')


if __name__ == '__main__':
    main(sys.argv[1:])
