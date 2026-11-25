import argparse
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory


def unpack(archive, destination):
    import py7zr

    with TemporaryDirectory(dir=destination.parent) as directory, py7zr.SevenZipFile(archive) as source:
        members = [name for name in source.getnames() if PurePosixPath(name).name == 'new_reddit.db']
        if len(members) != 1 or '\\' in members[0] or '..' in PurePosixPath(members[0]).parts:
            raise ValueError('Expected one safe new_reddit.db member')
        member = PurePosixPath(members[0])
        if member.is_absolute() or any(':' in piece for piece in member.parts):
            raise ValueError('Unexpected SQLite member path')
        source.extract(path=directory, targets=[members[0]])
        extracted = Path(directory).joinpath(*member.parts)
        with extracted.open('rb') as file:
            if file.read(16) != b'SQLite format 3\0':
                raise ValueError('Expected a SQLite database')
        extracted.replace(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    unpack(args.archive, args.destination)
