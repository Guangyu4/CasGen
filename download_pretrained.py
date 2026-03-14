"""Download pre-trained models to local pretrained/ directory.

Run this script on a machine WITH internet access, then copy the pretrained/
folder to the cluster.

Usage:
    python download_pretrained.py
    python download_pretrained.py --model bert-base-chinese
"""
import argparse
import os
from huggingface_hub import snapshot_download

MODELS = ['bert-base-uncased', 'bert-base-chinese']


def download(model_name, local_dir):
    out = os.path.join(local_dir, model_name)
    if os.path.exists(out) and os.listdir(out):
        print(f'Already exists: {out}')
        return
    print(f'Downloading {model_name} -> {out}')
    snapshot_download(repo_id=model_name, local_dir=out,
                      ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'])
    print(f'Done: {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None, help='Single model to download; omit for all')
    p.add_argument('--outdir', default='pretrained')
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    targets = [a.model] if a.model else MODELS
    for m in targets:
        download(m, a.outdir)


if __name__ == '__main__':
    main()
