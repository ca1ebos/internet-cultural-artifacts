import argparse
import os
import shutil
import time

from common import ARTIFACTS, CONFIG

FILES = ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt',
         'rng_state.pth', 'trainer_state.json')


def preserve():
    for source in CONFIG.keys() - {'base', 'seed'}:
        directory = ARTIFACTS / 'runs' / source
        for checkpoint in directory.glob('*/checkpoint-*'):
            if not all((checkpoint / name).is_file() for name in FILES):
                continue
            stage = checkpoint.parent.name
            dest = ARTIFACTS / 'preserved-checkpoints' / source / stage / checkpoint.name
            if (dest / 'complete').exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            try:
                for current in checkpoint.iterdir():
                    if not current.is_file():
                        continue
                    output = dest / current.name
                    pending = output.with_suffix(output.suffix + f'.{os.getpid()}.part')
                    pending.unlink(missing_ok=True)
                    try:
                        os.link(current, pending)
                    except OSError:
                        shutil.copyfile(current, pending)
                    pending.replace(output)
            except FileNotFoundError:
                continue
            if all((dest / name).is_file() for name in FILES):
                (dest / 'complete').touch()
                print(f'Preserved {source}/{stage}/{checkpoint.name}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    preserve()
    if args.watch:
        while not (ARTIFACTS / 'queue-completed.json').exists():
            time.sleep(5)
            preserve()
