import sys
sys.path.append('../')

import politician_news_analyzer
print('Available politician_news_analyzer == {}'.format(politician_news_analyzer.__version__))

import os
politician_news_dataset_path = '/mnt/sdc2/politician_news_dataset/'
if os.path.exists(politician_news_dataset_path):
    sys.path.append(politician_news_dataset_path)
    print('Available politician_news_dataset')
else:
    print('Check your carblog dataset or "git clone https://github.com/lovit/politician_news_dataset.git"')
    print('After cloning, you must install the dataset. See more politician_news_dataset README')
