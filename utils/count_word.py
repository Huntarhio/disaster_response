import re
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from collections import Counter
import pandas as pd
import numpy as np

def count_word(data):
    text = ''.join(data)
    new_text = re.sub('[^a-zA-Z]+', ' ', text)
    word_tokens = word_tokenize(''.join(new_text))

    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words] 
    
    counts = (Counter(filtered_sentence))
    counts = dict(counts)

    words = counts.keys()
    count = counts.values()
    counts_df = pd.DataFrame(zip(words,count), columns = ['words', 'counts'])
    
    counts_gr_500 = counts_df[counts_df['counts'] > 500]
    counts_list = list(counts_gr_500['counts'].values)
    words_list = list(counts_gr_500['words'].values)
    
    return counts_list, words_list