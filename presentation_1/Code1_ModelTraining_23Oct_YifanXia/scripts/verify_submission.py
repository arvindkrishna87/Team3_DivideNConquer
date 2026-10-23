"""Read-only verification of the exported scientific results, without training."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def require(condition,message):
    if not condition:
        raise RuntimeError(message)

def main():
    root=Path(__file__).resolve().parents[1]
    digests={}
    for line in (root/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        require(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,
                'File checksum mismatch: '+name)
        digests[name]=digest
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and p.name!='MANIFEST.sha256'}
    require(actual==set(digests),'Unexpected or missing submitted files')
    assignments=read_csv(root/'analysis/case_assignments.csv')
    universe={int(r['case_id']) for r in assignments}
    require(len(universe)==93,'Expected 93 unique cases')
    summary=json.loads((root/'results/five_split_summary/summary.json').read_text())
    combined=read_csv(root/'results/five_split_summary/out_of_fold_predictions.csv')
    metric_rows=read_csv(root/'results/five_split_summary/split_metrics.csv')
    test_counts=collections.Counter()
    pooled=np.zeros((2,2),dtype=int)
    accuracies=[]
    for split in range(1,6):
        folder=root/'results'/f'split{split:02d}'
        metric=json.loads((folder/'metrics.json').read_text())
        manifest=json.loads((folder/'run_manifest.json').read_text())
        require(manifest['status']=='COMPLETED','Incomplete run')
        require(manifest['seed']==3902026,'Seed mismatch')
        for name,digest in manifest['source_sha256'].items():
            require(hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==digest,
                    'Executed source mismatch: '+name)
        split_path=root/'splits'/f'data_splits_new_{split:02d}.npz'
        require(hashlib.sha256(split_path.read_bytes()).hexdigest()==manifest['split_sha256'],
                'Split checksum mismatch')
        sets={}
        families={}
        with np.load(split_path,allow_pickle=False) as saved:
            for subset in ('train','val','test'):
                selected=[r for r in assignments if int(r['split'])==split and r['subset']==subset]
                ids=list(map(int,saved[subset+'_cases']))
                sets[subset]=set(ids)
                require(len(ids)==len(sets[subset]),'Duplicate case within subset')
                require(sets[subset]=={int(r['case_id']) for r in selected},'Assignment mismatch')
                families[subset]={int(r['family_id']) for r in selected}
        require(set.union(*sets.values())==universe,'Incomplete split coverage')
        for a,b in (('train','val'),('train','test'),('val','test')):
            require(not sets[a]&sets[b],'Case leakage')
            require(not families[a]&families[b],'Family leakage')
        predictions=read_csv(folder/'predictions.csv')
        combined_split=[{k:v for k,v in r.items() if k!='split'} for r in combined if int(r['split'])==split]
        require(sorted(predictions,key=lambda r:int(r['case_id']))==
                sorted(combined_split,key=lambda r:int(r['case_id'])),'Combined probability mismatch')
        labels={int(r['case_id']):int(r['label']) for r in assignments
                if int(r['split'])==split and r['subset']=='test'}
        require(len(predictions)==len(labels),'Test count mismatch')
        require({int(r['case_id']) for r in predictions}==sets['test'],'Prediction IDs mismatch')
        matrix=np.zeros((2,2),dtype=int)
        for row in predictions:
            cid,y,pred=int(row['case_id']),int(row['true_label']),int(row['predicted_label'])
            require(y==labels[cid],'True label mismatch')
            probs=[float(row['prob_class0']),float(row['prob_class1'])]
            require(all(math.isfinite(p) and 0<=p<=1 for p in probs),'Invalid probability')
            require(math.isclose(sum(probs),1,abs_tol=1e-5),'Probabilities do not sum to one')
            require(pred==int(np.argmax(probs)),'Prediction differs from argmax')
            require(row['correct'].lower()==str(y==pred).lower(),'Incorrect correctness flag')
            matrix[y,pred]+=1
            test_counts[cid]+=1
        require(matrix.tolist()==metric['confusion_matrix_true_rows_predicted_columns'],
                'Confusion matrix mismatch')
        values={'accuracy':float(np.trace(matrix)/matrix.sum()),
                'benign_recall':float(matrix[0,0]/matrix[0].sum()),
                'high_grade_recall':float(matrix[1,1]/matrix[1].sum())}
        aggregate_row=next(r for r in metric_rows if int(r['split'])==split)
        for k,v in values.items():
            require(math.isclose(v,metric[k]),'Metric mismatch: '+k)
            require(math.isclose(v,float(aggregate_row[k])),'Summary metric mismatch: '+k)
        require(float(aggregate_row['main_execution_seconds'])==metric['main_execution_seconds'],
                'Runtime mismatch')
        history=read_csv(folder/'epoch_history.csv')
        require([int(r['epoch']) for r in history]==list(range(1,metric['epochs_completed']+1)),
                'Incomplete epoch history')
        accuracies.append(values['accuracy'])
        pooled+=matrix
    require(set(test_counts)==universe and all(v==1 for v in test_counts.values()),
            'Each case must be tested exactly once')
    require(pooled.tolist()==summary['pooled_out_of_fold']['confusion_matrix_true_rows_predicted_columns'],
            'Pooled confusion matrix mismatch')
    require(math.isclose(sum(accuracies)/5,summary['across_split_statistics']['accuracy']['mean']),
            'Mean accuracy mismatch')
    require(math.isclose(float(np.trace(pooled)/pooled.sum()),summary['pooled_out_of_fold']['accuracy']),
            'Pooled accuracy mismatch')
    print(json.dumps({'status':'PASS','files_verified':len(digests),'splits_verified':5,
                      'unique_test_cases':len(test_counts),'correct_predictions':int(np.trace(pooled)),
                      'mean_accuracy':sum(accuracies)/5,'training_started':False},indent=2))

if __name__=='__main__':
    main()
