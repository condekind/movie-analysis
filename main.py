#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ast
import numpy as np
from scipy.stats import iqr
from pprint import pprint
from datetime import datetime
from IPython.core.display import HTML
import re
import string

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '{:⎯>13.3f}'.format(x))

# plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, FormatStrFormatter
from matplotlib import cm
from colorspacious import cspace_converter

# fluff
from jupyterthemes import jtplot
jtplot.style()
from kindlib.fluff import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")


# In[ ]:


plt.rcParams['axes.labelsize']  = 12
plt.rcParams['axes.titlesize']  = 16
plt.rcParams['font.family']='monospace'
plt.rcParams['font.size']       = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


mdf = pd.read_pickle('data/movies_metadata.pkl')
rawdf = pd.read_pickle('data/movies_metadata.pkl')

pprint(list(mdf.columns))
mdf.shape


# ### <center>Caracterização</center>

# In[ ]:


colstats = Colstats(mdf)

print('Total number of entries:', len(mdf))
colstat = []
for idx, row in colstats.iterrows():
    df = pd.DataFrame(data=row).drop('col').rename(columns={idx:colstats['col'].iloc[idx]}).T
    df.index.name = 'Feature'
    if int(df['nunique']) <= 10:
        df2 = pd.DataFrame(mdf[row['col']].unique(), columns=['values'])
        p(df, df2)
    else:
        display(df)
    print('')


# In[ ]:


mdf.columns


# In[ ]:


typecol = {
    'json' : [
        'belongs_to_collection',
        'genres',
        'production_companies',
        'production_countries',
        'spoken_languages',
    ],
    'string' : [
        'imdb_id',
        'original_language',
        'original_title',
        'overview',
        'status',
        'tagline',
        'title',
    ],
    'url' : [
        'homepage',
    ],
    'datetime' : [
        'release_date',
    ],
    'int' : [
        'budget',
        'id',
        'revenue',
        'runtime',
        'vote_count'
    ],
    'float' : [
        'popularity',
        'vote_average',
    ],
    'bool' : [
        'adult',
    ],
    'other' : [
        'poster_path',
    ]
}
unused_cols = [
    'homepage',
    'original_language',
    'overview',
    'poster_path',
    'tagline',
    'video',
]

coltype = {c:k for k, v in typecol.items() for c in v if c not in unused_cols}
# drop unused columns
mdf = mdf.drop(columns=[c for c in mdf.columns if c not in coltype])
before = mdf.dtypes


# In[ ]:


# time conversion
for c in typecol['datetime']:
    print(len(mdf))
    timedf = pd.to_datetime(mdf[c].dropna(), errors='coerce').dropna().astype(np.datetime64)
    mdf = mdf.iloc[timedf.index]
    mdf[c] = timedf
    print(len(mdf))


# In[ ]:


# integer conversion
for c in typecol['int']:
    mdf[c] = pd.to_numeric(mdf[c], errors='coerce')
    mdf = mdf.dropna(subset=[c])
    mdf[c] = mdf[c].astype('int')


# In[ ]:


# float conversion
for c in typecol['float']:
    mdf[c] = pd.to_numeric(mdf[c], errors='coerce')
    mdf = mdf.dropna(subset=[c])
    mdf[c] = mdf[c].astype('float')


# In[ ]:


# bool conversion
for c in typecol['bool']:
    mdf[c] =  mdf[mdf[c] == 'False'].append(
        other=mdf[mdf[c] == 'True'], verify_integrity=True)[c] == 'True'


# Unreliable columns: <br><br>
# 
# | Dropped columns       | Reason                       | Alternative                              |
# | --------------------- |:----------------------------:| ----------------------------------------:|
# | belongs_to_collection | too many nans                | &nbsp;&nbsp;convert to bool              |
# | homepage              | too many nans                | &nbsp;&nbsp;convert to bool              |
# | runtime               | estimation pollutes analysis | &nbsp;&nbsp;use mean or median of others |
# 
# <br>Let's drop them:

# In[ ]:


mdf = mdf.drop(columns=['belongs_to_collection', 'runtime'])  # homepage was already dropped


# In[ ]:


# removing invalid dupes
print('Shape before removing dupes:', mdf.shape)
mdf = mdf.drop(index=mdf[mdf['id'].duplicated() | mdf['imdb_id'].duplicated()].index)
print('Shape after removing dupes:', mdf.shape)

# removing entries with zero budget or zero revenue
moneydf = mdf[mdf['budget'].apply(lambda x: bool(x))]
print('Number of movies with budget info:', moneydf.shape[0])

moneydf = mdf[mdf['revenue'].apply(lambda x: bool(x))]
print('Number of movies with revenue info:', moneydf.shape[0])

moneydf = mdf[['budget','revenue']]
moneydf = moneydf[moneydf['budget'].apply(lambda x: bool(x)) & moneydf['revenue'].apply(lambda x: bool(x))]
print('Number of movies with budget and revenue info:', moneydf.shape[0])

moneydf = mdf.loc[moneydf.index]


# In[ ]:



colstats = Colstats(moneydf)
for idx, row in colstats.iterrows():
    df = pd.DataFrame(data=row).drop('col').rename(columns={idx:colstats['col'].iloc[idx]}).T
    df.index.name = 'Feature'
    if int(df['nunique']) <= 10:
        df2 = pd.DataFrame(moneydf[row['col']].unique(), columns=['values'])
        p(df, df2)
    else:
        display(df)
    print('')
moneydf[['budget', 'popularity', 'revenue', 'vote_average']].describe().round(3)


# <br><br>Now that our base is looking good for the basic types, let's try some plotting:

# In[ ]:


bw_methods = ['scott', 'silverman'] #, 0.1, 0.25, 0.5, 0.75, 1.0]
for c in moneydf[['budget', 'popularity', 'revenue', 'vote_average']].columns:
    fig, ax = plt.subplots()
    line = {}
    for idx, bw in enumerate(bw_methods):
        line[idx] = moneydf[c].plot.kde(bw_method=bw)
        line[idx].set_xlim(
            xmin=moneydf[c].quantile(0.25)-iqr(moneydf[c].values)*1.5,
            xmax=moneydf[c].quantile(0.75)+iqr(moneydf[c].values)*1.5)
        max_exp = int(np.floor(np.log10(moneydf[c].quantile(0.75)+iqr(moneydf[c].values)*1.5)))
        line[idx].ticklabel_format(axis='x', style='sci', scilimits=(max_exp,max_exp))
        line[idx].set_title(c, fontdict={'y': 0})
        line[idx].set_label(bw)
    plt.legend(bw_methods)
    plt.show()
    plt.close()
pass


# <hr>
# 
# Genres

# In[ ]:


# possibly interesting data
genre_data = {}

print('Total entries so far:', len(moneydf))

# ignoring possibility of genre id collision
moneydf.apply(axis=1,
        func=lambda x:
        [
            genre_data.update({k: v})
            for (_, k), (_, v) in
            [
                g.items()
                    if len(g) > 0
                    else {}
                for g in ast.literal_eval(x['genres'])
            ]
        ]
)

print('')
print('Number of distinct genres:', len(genre_data))
pprint(genre_data)


# <hr>

# ### <center>Retorno Financeiro e Gêneros</center>

# In[ ]:


# relevant info for H0
gdf = moneydf[['title', 'budget', 'revenue', 'genres', 'release_date']]

# creating columns to indicate genre
for _, g in genre_data.items():
    gdf = gdf.assign(**{str(g) : lambda _: [0 for _ in range(gdf.shape[0])]})

# setting correct genre values to 1 on each movie
genres = list(genre_data.values())
def set_genres(row:pd.Series, ginfo):
    genre_series = pd.Series(np.zeros(len(row)), index=row.index, dtype=int)
    for genre_info in ast.literal_eval(ginfo):
        genre_series[genre_info['name']] = 1
    return genre_series

# updating genre df
gdf[genres] = gdf[['genres']+genres].apply(axis=1, func=lambda row: set_genres(row, row['genres'])).drop(columns=['genres'])
gdf = gdf.drop(columns=['genres'])
gdf['year'] = gdf['release_date'].apply(func=lambda x: x.year)
gdf[genres].head(5)


# In[ ]:


# value correction -> real value = nominal value / gdp deflator 
price_deflator_df = pd.read_csv('data/GDPCTPI.csv').sort_values(by=['DATE'])
price_deflator_df['DATE'] = price_deflator_df['DATE'].astype(np.datetime64)
price_deflator_df = price_deflator_df.set_index(keys=['DATE'], verify_integrity=True)
price_deflator_df['y'] = price_deflator_df.apply(axis=1, func=lambda x: x.name.year)
price_deflator_df = pd.DataFrame(price_deflator_df.groupby(by=['y'])['GDPCTPI'].mean())
assert(len(price_deflator_df) == max(price_deflator_df.index) - min(price_deflator_df.index) + 1)

gdf = gdf.drop(gdf[gdf['year'] < min(price_deflator_df.index)].index)
gdf['gdpctpi'] = gdf['year'].apply(lambda x: price_deflator_df.loc[x])
gdf['budget_adjusted'] = gdf['budget'] / gdf['gdpctpi']
gdf['revenue_adjusted'] = gdf['revenue'] / gdf['gdpctpi']
gdf['profit_2012'] = (gdf['revenue_adjusted'] - gdf['budget_adjusted']).round(2)


# In[ ]:


gdf = gdf[['title', 'budget', 'budget_adjusted', 'revenue', 'revenue_adjusted', 'year',
        'profit_2012', 'gdpctpi', 'release_date', 'Animation', 'Comedy', 'Family',
        'Adventure', 'Fantasy', 'Drama', 'Romance', 'Action', 'Crime',
        'Thriller', 'History', 'Science Fiction', 'Mystery', 'Horror', 'War',
        'Foreign', 'Documentary', 'Western', 'Music', 'TV Movie']]


# In[ ]:


gdf.head(0)


# ### Let's see the big winners and big losers!

# In[ ]:


p(
    mdf.loc[gdf.sort_values(by=['profit_2012'])[-5:].index][['title']],
   gdf.sort_values(by=['profit_2012'])[-5:][['profit_2012']]
)
p(
    mdf.loc[gdf.sort_values(by=['profit_2012'], ascending=False)[-5:].index][['title']],
   gdf.sort_values(by=['profit_2012'], ascending=False)[-5:][['profit_2012']]
)
pass


# In[ ]:


bw_methods = ['scott', 'silverman'] #, 0.1, 0.25, 0.5, 0.75, 1.0]
for c in gdf[['budget_adjusted', 'revenue_adjusted', 'profit_2012', 'year']].columns:
    fig, ax = plt.subplots()
    line = {}
    for idx, bw in enumerate(bw_methods):
        line[idx] = gdf[c].plot.kde(bw_method=bw)
        line[idx].set_xlim(
            xmin=gdf[c].quantile(0.25)-iqr(gdf[c].values)*1.5,
            xmax=gdf[c].quantile(0.75)+iqr(gdf[c].values)*1.5)
        max_exp = int(np.floor(np.log10(gdf[c].quantile(0.75)+iqr(gdf[c].values)*1.5)))
        if c != 'year':
            line[idx].ticklabel_format(axis='x', style='sci', scilimits=(max_exp,max_exp))
        else:
            line[idx].ticklabel_format(axis='x', style='plain')
        line[idx].set_title(c, fontdict={'y': 0})
        line[idx].set_label(bw)
    plt.legend(bw_methods)
    plt.show()
    plt.close()
pass


# In[ ]:


# double checking if edge values make sense
p(
    gdf.sort_values(by=['profit_2012'], ascending=False)[['title', 'profit_2012', 'release_date']].head(5),
    gdf.sort_values(by=['profit_2012'], ascending=False)[['title', 'profit_2012', 'release_date']].tail(5)
)
pass


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
ax.yaxis.set_major_formatter(FormatStrFormatter('%{}.0f'.format(len(str(max(gdf['profit_2012'])))-3)))
ax.scatter(x=gdf['year'], y=gdf['profit_2012'], c=gdf['profit_2012']**(1/5))
ax.set_xlabel('Year')
ax.set_ylabel('Revenue - Budget (adjusted)')
plt.title('Real profit over the years')
plt.show()


# In[ ]:


gdf2 = pd.DataFrame([gdf[c].value_counts() for c in genres]).sort_values(by=1, ascending=False)

# genre presence in movies, percentage
gdf2['pmovie_pct'] = gdf2[1]/len(gdf)

# genre presence amongst other genres, percentage
gdf2['pgenre_pct'] = gdf2[1]/sum(gdf2[1])

# sum of adjusted profit and loss for all movies tagged with each genre
gdf2 = gdf2.assign(**{'genre_profit': [0 for f in range(len(gdf2))]})
gdf2 = gdf2.assign(**{'genre_losses': [0 for f in range(len(gdf2))]})


# movies can have more than one tag and will contribute to multiple genres
for c in genres:
    profit = gdf[gdf['profit_2012'] > 0].groupby(by=[c])['profit_2012'].sum()
    losses = gdf[gdf['profit_2012'] < 0].groupby(by=[c])['profit_2012'].sum()
    
    if 1 in profit.index:
        gdf2.loc[c, 'genre_profit'] = profit.loc[1]
    if 1 in losses.index:
        gdf2.loc[c, 'genre_losses']   = losses.loc[1]


# In[ ]:


p(
    pd.DataFrame(gdf2['pmovie_pct'].apply(lambda c: '{:.3%}'.format(c))),
    pd.DataFrame(gdf2['pgenre_pct'].apply(lambda c: '{:.3%}'.format(c))),
    pd.DataFrame(gdf2['genre_profit'].astype(int).apply(lambda c: '{:12d}'.format(c))),
    pd.DataFrame(gdf2['genre_losses'].astype(int).apply(lambda c: '{:12d}'.format(c)))
)
pass


# In[ ]:


fig = plt.figure(figsize=(19, 15))
plt.matshow(gdf[genres].corr(), fignum=fig.number, cmap='YlGn')
plt.xticks(range(gdf[genres].shape[1]), gdf[genres].columns, fontsize=14, rotation=45)
plt.yticks(range(gdf[genres].shape[1]), gdf[genres].columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# In[ ]:


#def gcolor(df):

#for c in gdf[['budget_adjusted', 'revenue_adjusted', 'profit_2012']].columns:
#    fig, ax = plt.subplots()
#    Y = gdf[gdf[c].between(gdf[c].quantile(0.25), gdf[c].quantile(0.75))]
#    ax.set_xlim(xmin=min(Y['year']), xmax=max(Y['year']))
#    ax.scatter(x=Y['year'], y=Y[c], c=Y[genres].apply(gcolor), cmap='rainbow')
#    ax.yaxis.set_major_formatter(FormatStrFormatter('%{}.0f'.format(len(str(max(Y['profit_2012'])))-3)))
#    ax.set_title(c)
#    plt.show()
#    plt.close()
#pass


# #### II. H<sub>0</sub>: A média de idade dos 10 atores principais está correlacionada com os ratings.

# <hr>

# ### <center>Regressão</center>

# In[ ]:


mdf['popularity'].describe()


# Vamos considerar como popular os filmes com um valor de 'popularity' maior que 2 (um pouco menor que a média)

# In[ ]:


new_mdf = mdf
new_mdf['popularity'] = (new_mdf['popularity'] > 2).astype(int)


# In[ ]:


new_mdf['popularity'].value_counts()


# In[ ]:


X = new_mdf[['vote_average', 'vote_count']]
y = new_mdf['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

accuracy = logistic_regression.score(X_test, y_test)
print(accuracy)


# A acurácia foi de aproximadamente 90%

# In[ ]:


print(X_test)


# In[ ]:


# testando com outros valores
new_votes = {'vote_average': [6.7, 5.8, 3.6, 7.7, 9.8, 8.9],
            'vote_count': [10, 30, 49, 88, 19, 70]}

df = pd.DataFrame(new_votes, columns=['vote_average', 'vote_count'])
y_pred = logistic_regression.predict(df)
print(df)
print(y_pred)


# In[ ]:


# agora considerando o revenue do filme
new_mdf = moneydf
new_mdf['popularity'] = (new_mdf['popularity'] > 2).astype(int)
X = new_mdf[['vote_average', 'vote_count', 'revenue']]
y = new_mdf['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

accuracy = logistic_regression.score(X_test, y_test)
print(accuracy)


# A acurácia foi para aproximadamente 91%

# ### <center>Classificação</center>

# #### I. H<sub>0</sub>: O gênero do filme impacta diretamente em sua rentabilidade

# #### Random Forest

# In[ ]:


from sklearn.svm import SVC#### I. H<sub>0</sub>: O revenue de filmes do mesmo gênero não é influenciado pela data de lançamento.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split


# In[ ]:


gdf['popularity'] = mdf.loc[gdf.index]['popularity']
features = genres+['popularity', 'year']

min_threshhold = gdf['profit_2012'].quantile(0.7)
max_threshhold = gdf['profit_2012'].quantile(0.95)

gdf['isrich'] = (gdf['profit_2012'] > min_threshhold) & (gdf['profit_2012'] < max_threshhold)


# In[ ]:


X, y = gdf[features].values, gdf['isrich'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)


# In[ ]:


svc_disp = plot_roc_curve(svc, X_test, y_test)
plt.show()


# In[ ]:


rfc = RandomForestClassifier(n_estimators=128, random_state=99, criterion='gini')
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()


# In[ ]:


rfc = RandomForestClassifier(n_estimators=128, random_state=99, criterion='entropy')
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()


# ### <center>Relatório Final:</center>
# <br>
# 
# #### Sua análise dos dados deve apresentar:
# - Uma caracterização (análise exploratória) inicial dos dados **(2pts)**
# - Pelo menos, dois testes de hipótese/intervalos de confiança **(2pts)**
#     - Os ICs podem ser apresentados nos resultados de regressão e classificação abaixo.
#     - Os testes de hipótese também podem ser utilizados abaixo para comparar modelos.
# - Pelo menos uma regressão **(3pts)**
# - Pelo menos um algoritmo de aprendizado/classificação **(3pts)**
# 
# #### No seu relatório, você deve apresentar pelo menos os seguintes pontos:
# - Introdução com Motivação e Pergunta de Pesquisa
# - Metodologia
# - Descreva sua base
# - Quais métodos e modelos foram utilizados. Justifique os mesmos.
# - Resultados. Sugiro separar em
#     - Caracterização (análise exploratória)
#     - Testes de hipótese podem vir aqui.
#     - Previsão (uma ou duas sub-seções dependendo dos modelos utilizados)
#     - Conclusões
# 
# Responda suas perguntas:
# - Qual a melhor época do ano para anunciar e lançar um filme?
# - Como a popularidade dos gêneros dos filmes evoluiu ao longo dos anos?
# - Qual o peso de um ator/atriz popular no retorno financeiro de um filme com avaliação "ruim"?
# 
# ### Vídeo
# Vídeo no Youtube 5 minutos (pode ser um vídeo só de slides) **(5pts)**
