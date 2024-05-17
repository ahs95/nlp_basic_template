import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk

from tqdm import trange
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')
nltk.download('omw-1.4', quiet=True)
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (17,7)
plt.rcParams['font.size'] = 18

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
import torch.nn as nn
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import classification_report, confusion_matrix, f1_score

data = pd.read_csv('csv_file_name')
data

data.info()
data['column_name'].value_counts()
