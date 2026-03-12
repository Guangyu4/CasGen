"""Convert processed.pt to EdiTPP format .pkl"""
import os
import torch

PROCESSED = os.path.join(os.path.dirname(__file__), 'dataset', 'processed.pt')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'editpp', 'data')
OUT_PATH = os.path.join(OUT_DIR, 'weibo_cascade.pkl')


def main():
    data = torch.load(PROCESSED, map_location='cpu', weights_only=False)
    cascades = data['cascades']
    stats = data['stats']

    sequences = []
    for c in cascades:
        times = c['times'].tolist()
        if len(times) > 0:
            sequences.append({'arrival_times': times})

    t_max = 1.0

    out = {'sequences': sequences, 't_max': t_max}
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(out, OUT_PATH)
    print(f'Saved {len(sequences)} sequences to {OUT_PATH}')
    print(f't_max={t_max}')

    lens = [len(s['arrival_times']) for s in sequences]
    print(f'Lengths: min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.1f}')


if __name__ == '__main__':
    main()
