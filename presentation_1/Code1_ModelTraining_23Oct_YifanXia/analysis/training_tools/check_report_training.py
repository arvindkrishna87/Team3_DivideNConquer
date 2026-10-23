from pathlib import Path
import contextlib,csv,io,re,tempfile
import numpy as np
from report_training import summarize
CSV_TEXT='case_id,true_label,predicted_label,correct,prob_class0,prob_class1\n9,1,0,False,0.9954929351806641,0.004507023375481367\n16,1,1,True,0.11687775701284409,0.8831222653388977\n26,0,1,False,0.2659945487976074,0.7340054512023926\n32,1,0,False,0.8513457775115967,0.14865419268608093\n40,1,1,True,4.387066383060301e-06,0.9999955892562866\n45,0,1,False,0.1647195965051651,0.8352803587913513\n50,1,1,True,0.03407697752118111,0.9659230709075928\n55,1,1,True,3.054136143987307e-08,1.0\n60,1,1,True,0.03478666767477989,0.9652132987976074\n67,1,1,True,0.00038757111178711057,0.9996125102043152\n73,1,1,True,0.0018614677246659994,0.9981385469436646\n82,0,0,True,0.9994472861289978,0.000552649493329227\n86,0,0,True,0.7038196325302124,0.2961803376674652\n91,1,1,True,0.012223909609019756,0.9877760410308838\n99,1,1,True,3.235558097003377e-06,0.9999967813491821\n101,1,1,True,0.00540558947250247,0.994594395160675\n109,0,0,True,0.9977795481681824,0.0022204627748578787\n110,0,0,True,0.9998001456260681,0.0001998148363782093\n111,0,0,True,0.9861176609992981,0.013882330618798733\n'
RESULTS_TEXT='Test Results:\nTest Loss: 0.5799\nTest Accuracy: 0.7895\nNumber of samples: 19\n\nOutput files:\n- predictions.csv: Per-case predictions and probabilities\n- confusion_matrix.png: Visual confusion matrix\n- attention_analysis/: Attention visualizations and summary\n'
LOG_TEMPLATE='Created run directory: /projects/e32998/STAT390_Fall2026/Team3_DivideNConquer/presentation_1/taylor_runs/run_20261020_180131\nEpoch 1/30 (LR: 2.08e-04)\nTrain Loss: 1.3047 | Train CE: 1.3057 | Train Patch Entropy: 0.9959 | Val Loss: 0.5044 | Val Acc: 0.7222\nEpoch Time: 273.19s\nEpoch 2/30 (LR: 2.08e-04)\nTrain Loss: 0.4604 | Train CE: 0.4614 | Train Patch Entropy: 0.9808 | Val Loss: 0.2411 | Val Acc: 0.8889\nEpoch Time: 148.35s\nEpoch 3/30 (LR: 2.08e-04)\nTrain Loss: 0.2403 | Train CE: 0.2412 | Train Patch Entropy: 0.9525 | Val Loss: 0.3289 | Val Acc: 0.8889\nEpoch Time: 51.79s\nEpoch 4/30 (LR: 2.08e-04)\nTrain Loss: 0.1802 | Train CE: 0.1811 | Train Patch Entropy: 0.9247 | Val Loss: 0.2633 | Val Acc: 0.8889\nEpoch Time: 53.27s\nEpoch 5/30 (LR: 2.08e-04)\nTrain Loss: 0.0532 | Train CE: 0.0541 | Train Patch Entropy: 0.9231 | Val Loss: 0.2893 | Val Acc: 0.8889\nEpoch Time: 50.18s\nEpoch 6/30 (LR: 2.08e-04)\nTrain Loss: 0.0410 | Train CE: 0.0420 | Train Patch Entropy: 0.9258 | Val Loss: 0.2694 | Val Acc: 0.7778\nEpoch Time: 84.06s\nEpoch 7/30 (LR: 2.08e-04)\nTrain Loss: 0.2437 | Train CE: 0.2446 | Train Patch Entropy: 0.8997 | Val Loss: 0.4847 | Val Acc: 0.8333\nEpoch Time: 51.48s\nEpoch 8/30 (LR: 4.45e-05)\nTrain Loss: 0.0839 | Train CE: 0.0848 | Train Patch Entropy: 0.9120 | Val Loss: 0.3717 | Val Acc: 0.8889\nEpoch Time: 52.02s\nEpoch 9/30 (LR: 4.45e-05)\nTrain Loss: 0.0176 | Train CE: 0.0185 | Train Patch Entropy: 0.9122 | Val Loss: 0.3032 | Val Acc: 0.8889\nEpoch Time: 51.27s\nEpoch 10/30 (LR: 4.45e-05)\nTrain Loss: 0.0139 | Train CE: 0.0148 | Train Patch Entropy: 0.9110 | Val Loss: 0.3368 | Val Acc: 0.8889\nEpoch Time: 50.17s\nTotal execution time: 1148.64s (0.32 hours)\nTraining completed successfully!'

with tempfile.TemporaryDirectory(prefix='yifan-report-check-') as temp:
    base=Path(temp);run=base/'runs/split02/reference_fixture';(run/'checkpoints').mkdir(parents=True)
    (base/'splits').mkdir()
    (run/'predictions.csv').write_text(CSV_TEXT);(run/'results.txt').write_text(RESULTS_TEXT)
    rows=list(csv.DictReader(io.StringIO(CSV_TEXT)))
    np.savez(base/'splits/data_splits_new_02.npz',test_cases=np.array([int(r['case_id']) for r in rows]))
    old='/projects/e32998/STAT390_Fall2026/Team3_DivideNConquer/presentation_1/taylor_runs/run_20261020_180131'
    log=LOG_TEMPLATE.replace(old,str(run));(base/'training.log').write_text(log)
    last=max(int(x) for x in re.findall(r'^Epoch (\d+)/',log,re.MULTILINE))
    (run/'checkpoints'/f'fixture_epoch{last}.pth').touch();(run/'confusion_matrix.png').touch()
    with contextlib.redirect_stdout(io.StringIO()):
        metrics=summarize(base,2,base/'training.log',base/'analysis')
    assert metrics['correct']==15 and metrics['n_test_cases']==19
    assert abs(metrics['benign_recall']-5/7)<1e-12
    assert abs(metrics['high_grade_recall']-10/12)<1e-12
    assert metrics['epochs_completed']==last
    assert (base/'analysis/loss_curves.png').stat().st_size>1000
    (run/'predictions.csv').write_text(CSV_TEXT.replace('9,1,0,False','9999,1,0,False',1))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            summarize(base,2,base/'training.log',base/'bad_analysis')
    except RuntimeError as error:
        assert 'IDs' in str(error)
    else:
        raise AssertionError('Wrong test cases were not rejected')
print('Reporting self-check passed: known metrics, epoch parsing, plot generation, wrong-test-set rejection.')
