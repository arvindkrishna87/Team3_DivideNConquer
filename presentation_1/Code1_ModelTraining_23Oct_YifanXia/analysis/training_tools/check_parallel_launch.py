"""Check that simultaneous runners produce distinct baseline run directories."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import sys
import tempfile

from run_training import run_logged_process, build_main_command


def main():
    workspace = Path(sys.argv[1])
    tree = ast.parse((workspace / 'code/utils.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'create_run_directory')
    actual_function = ast.unparse(function)
    with tempfile.TemporaryDirectory(prefix='launch_check_') as directory:
        root = Path(directory)
        (root / 'runs').mkdir()
        (root / 'config.py').write_text(
            'DATA_PATHS = ' + repr({'runs_dir': str(root / 'runs')}) + '\n')
        (root / 'main.py').write_text(
            'import os\n' + actual_function + '\ncreate_run_directory()\n')

        def launch(index):
            log = root / f'{index}.log'
            parent = root / 'runs' / f'split{index:02d}'
            parent.mkdir()
            command = build_main_command(sys.executable, root, parent, [])
            result = run_logged_process(command, root, log)
            assert result == 0
            match = re.search(r'Created run directory: (.+)', log.read_text())
            assert match
            assert Path(match.group(1)).parent == parent
            return match.group(1)

        with ThreadPoolExecutor(max_workers=4) as pool:
            runs = list(pool.map(launch, range(4)))
        assert len(set(runs)) == 4, runs
        assert all(Path(run).is_dir() for run in runs)
        assert len(list((root / 'runs').glob('split*/run_*'))) == 4
        assert 'runs_dir' in (root / 'config.py').read_text()
        assert (root / 'config.py').read_text() == 'DATA_PATHS = ' + repr({'runs_dir': str(root / 'runs')}) + '\n'
    print('Output isolation self-check passed: four simultaneous requests created runs under distinct split-specific parents without modifying config.py.')


if __name__ == '__main__':
    main()
