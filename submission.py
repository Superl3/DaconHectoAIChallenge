import pandas as pd
from utils import load_config
from infer import inference

def make_submission():
    cfg = load_config()
    pred, class_names = inference()
    submission = pd.read_csv(cfg['sample_submission'], encoding='utf-8-sig')
    class_columns = submission.columns[1:]
    pred = pred[class_columns]
    submission[class_columns] = pred.values
    submission.to_csv('baseline_submission.csv', index=False, encoding='utf-8-sig')
    print('Submission file saved as baseline_submission.csv')

if __name__ == '__main__':
    make_submission()
