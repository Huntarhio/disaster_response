import re
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from collections import Counter
import pandas as pd
import numpy as np

import re
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from collections import Counter
import pandas as pd
import numpy as np

def count_word(data):
    """
    this function counts the words in the all messages in the data and returns the most common words

    PARAMETER:
        data (series): the series of all messages in the data

    RETURN:
        count_list (list): list of the counts of the most common words
        words_list (list): list of the most common words
    """
    text = ''.join(data)  # joining all the rows into one
    new_text = re.sub('[^a-zA-Z]+', ' ', text)  # extracting on the words
    word_tokens = word_tokenize(''.join(new_text))  #tokenizing into items


    # removing all stopwords 
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words] 
    
    counts = (Counter(filtered_sentence))  # counting the words and saving to dictionary
    counts = dict(counts)

    words = counts.keys()
    count = counts.values()
    counts_df = pd.DataFrame(zip(words,count), columns = ['words', 'counts'])
    
    counts_gr_500 = counts_df[counts_df['counts'] > 500]
    counts_list = list(counts_gr_500['counts'].values)
    words_list = list(counts_gr_500['words'].values)
    
    return counts_list, words_list