# %%
import numpy as np 
import pandas as pd
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import pyLDAvis as plda
import pyLDAvis.gensim as plgen
import re
import gensim.corpora as gcp
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.hdpmodel import HdpModel
import nlpaug.augmenter.word as naw


# %% [markdown]
# ## 1. Data Aquisition

# %%
df = pd.read_excel("merged_data-v1.1.xlsx")
df

# %% [markdown]
# # 1.2 EDA

# %%
df["normTitle"].unique()

# %%
df.describe()

# %%
df.isna().sum()

# %%
df.columns

# %%
df_column_clean = df.drop(["companyOverviewLink","viewJobLink",'companyRating', 'companyReviewCount',"taxonomyAttributes/1/attributes/0/label"],axis=1).copy()
df_column_clean.head()

# %% [markdown]
# # 2. Text Pre-processing

# %%
df_word_column = df["snippet"].copy().to_list()
df_word_column[:10]

# %% [markdown]
# # 3. Data Augmentation

# %%
df_randomize = pd.read_excel("merged_data-v1.1.xlsx").sample(n= int(np.floor(len(df)*0.1))).reset_index(drop=True)
snippet_info = naw.synonym.SynonymAug(aug_src='wordnet')
synonym_comments = [snippet_info.augment(sentence)[0] for sentence in df_randomize["snippet"].to_list()]

df_augment = df_randomize.copy()
df_augment["snippet"] = synonym_comments


len(df_augment)

# %%
df_merged = pd.concat([df,df_augment]).reset_index(drop=True)
df_merged

# %% [markdown]
# # 3.1 Text Processing Part 2

# %%
df_word_comments = [re.split(r"[,\s\t;.\n/]+",sentence) for sentence in df_merged["snippet"].copy().to_list()]
df_word_comments_cleaned = list()
for sentence in df_word_comments:
    df_word_comments_cleaned.append([word.lower() for word in sentence if word != ''])
df_word_comments

# %% [markdown]
# # 4. Feature Engineering

# %%
id2word = gcp.Dictionary(df_word_comments_cleaned)
corpus = [id2word.doc2bow(text) for text in df_word_comments]
corpus

# %%
corpus[:1]

# %%
[[(id2word[i], freq) for i, freq in doc] for doc in corpus[:3]]

# %% [markdown]
# # Creating ML model

# %%
lda_modoel = LdaModel(corpus=corpus,
                      id2word=id2word,
                      num_topics=20,
                      chunksize=100,
                      alpha='auto',
                      per_word_topics=True)
print(lda_modoel.print_topics())
doc_lda = lda_modoel[corpus]



''

# %%
plda.enable_notebook()
p = plgen.prepare(lda_modoel, corpus, id2word)
p


# %%
p.topic_info[p.topic_info["Category"] == "Topic1"]["Term"].to_list()

# %%
coherence_model_lda = CoherenceModel(model=lda_modoel, texts=df_word_comments_cleaned, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)

# %%
df["snippet_list"] = [re.split(r"[,\s\t;.\n]+",sentence) for sentence  in df["snippet"]].copy()

# %%
topics = p.topic_info[p.topic_info["Category"] == "Topic1"]["Term"].to_list()
remove_words = ['…',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'through',
    'if',
    'as',
    '*',
    'have',
    'course',
    'at',
    'are',
    'for…',
    '(40%)',
    'analysis?',
    'you',
    'that',
    'a',
    'in',
    'with',
    'and',
    'of',
    'to',
    'is',
    'be',
    'by',
    'will',
    'any',
    'on',
    'for',
    'the',
    'with…']
to_check = [word for word in topics if word not in remove_words]
to_check

# %%
df["extractedSalary/type"].unique()

# %%
df.columns

# %%
df_info = df.loc[:,['extractedSalary/max', 'extractedSalary/min', 'extractedSalary/type', 'snippet']].copy()
df_info.head()

# %%
df_info["snippet_list"] = [re.split(r"[,\s\t;.\n]+",sentence) for sentence  in df_info["snippet"]].copy()
df_info.head()

# %%
sentence_list = list()
item_topic_bool= list()
for array in df_info["snippet_list"]:
    topic_list_arr = list()
    item_count = 0
    for items in array:
        if items in to_check:
            topic_list_arr.append(True)
        else:
            topic_list_arr.append(False)
    if True in topic_list_arr:
        sentence_list.append(np.sum(topic_list_arr))
        item_topic_bool.append(True)
    else:
        sentence_list.append(np.sum(topic_list_arr))
        item_topic_bool.append(False)
        

# %%
df_info["Topic1?"] = item_topic_bool
df_info["Topic1Count"] = sentence_list

# %%
df_info

# %%
print("the key words are the ff ",to_check)
df_info.drop(["snippet","snippet_list"],axis=1).groupby(["Topic1?","Topic1Count","extractedSalary/type"]).mean()


