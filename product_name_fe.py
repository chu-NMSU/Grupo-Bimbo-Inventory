"""
product name feature extraction
"""

import pandas
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("spanish")

products  =  pandas.read_csv("data/producto_tabla.csv")
# extract fields. () means a regular expression group, it will only return the matched group
products['short_name'] = products.NombreProducto.str.extract('^(\D*)', expand=False)
products['brand'] = products.NombreProducto.str.extract('^.+\s(\D+) \d+$', expand=False)
w = products.NombreProducto.str.extract('(\d+)(Kg|g)', expand=True)
products['weight'] = w[0].astype('float')*w[1].map({'Kg':1000, 'g':1})
products['pieces'] =  products.NombreProducto.str.extract('(\d+)p ', expand=False).astype('float')
# remove stop words
products['short_name_processed'] = (products['short_name'].map(\
      lambda x:' '.join([i for i in x.lower().split() if i not in stopwords.words("spanish")])))
# stemming
products['short_name_processed'] = (products['short_name_processed']\
        .map(lambda x: ' '.join([stemmer.stem(i) for i in x.lower().split()])))
products.to_csv('data/producto_tabla_stem.csv', index=False)

# w_couter = Counter()
# with open('data/producto_tabla.csv') as infile:
#     head_flag = True
#     for line in infile:
#         if head_flag:
#             head_flag = False
#             continue
#         fields = line.strip().split(',')
#         p_name = fields[1]
#         tokens = p_name.split()
#         for w in tokens:
#             if w[0].isdigit():
#                 break
#             w_couter[w.lower()] += 1
#         # print p_name
# 
