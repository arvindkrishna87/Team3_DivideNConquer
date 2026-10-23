"""Prepare a new reproduction workspace without touching completed experiments."""
import argparse
import json
import shutil
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[1]
    target=args.workspace.expanduser().resolve()
    if target.exists():
        parser.error('Destination already exists; choose a fresh workspace.')
    target.mkdir(parents=True)
    for name in ('analysis','logs','runs','splits'):
        (target/name).mkdir()
    shutil.copytree(source/'code',target/'code')
    shutil.copytree(source/'analysis/training_tools',target/'analysis/training_tools')
    shutil.copy2(source/'analysis/validate_splits.py',target/'analysis/validate_splits.py')
    config=target/'code/config.py'
    text=config.read_text()
    old="'runs_dir': '/home/siw1333/Yifan_Xia_coder_p1/runs'"
    if text.count(old)!=1:
        raise RuntimeError('Unexpected source configuration; no replacement performed.')
    config.write_text(text.replace(old,"'runs_dir': "+repr(str(target/'runs'))))
    (target/'REPRODUCTION_SETUP.json').write_text(json.dumps({
        'submission_source':str(source),'workspace':str(target),
        'configuration_change':'New copy only: runs_dir redirected to reproduction workspace.',
        'training_started':False},indent=2)+'\n')
    print('Prepared '+str(target)+'. No training has started.')

if __name__=='__main__':
    main()
