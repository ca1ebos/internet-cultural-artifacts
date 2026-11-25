import argparse
from train import train
from common import CONFIG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=[name for name in CONFIG if name not in {'base', 'seed'}])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train(args.source, args.smoke, args.resume, None, 'auto')
