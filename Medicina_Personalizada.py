#!/usr/bin/env python
# coding: utf-8

# # 1. Problema de Negócio

# Muito tem sido dito durante os últimos anos sobre como a medicina de precisão e, mais concretamente, como o teste genético, vai provocar disrupção no tratamento de doenças como o câncer.
# 
# Mas isso ainda está acontecendo apenas parcialmente devido à enorme quantidade de trabalho manual ainda necessário. Neste projeto, tentaremos levar  a medicina personalizada ao seu potencial máximo. Uma vez sequenciado, um tumor cancerígeno pode ter milhares de mutações genéticas. O desafio é distinguir as mutações que contribuem para o  crescimento do tumor das mutações.
# 
# Atualmente, esta interpretação de mutações genéticas está sendo feita  manualmente. Esta é uma tarefa muito demorada, onde um patologista clínico tem  que revisar manualmente e classificar cada mutação genética com base em  evidências da literatura clínica baseada em texto.
# 
# Para este projeto, o MSKCC (Memorial Sloan Kettering Cancer Center) está  disponibilizando uma base de conhecimento anotada por especialistas, onde pesquisadores e oncologistas de nível mundial anotaram manualmente milhares de mutações.
# 
# Dataset: https://www.kaggle.com/c/msk-redefining-cancer-treatment/overview
# 
#     -- Objetivos
#         - Atingir 65% de precisão.
#         - Log Loss menor que 1.0 .

# # 2. Imports

# In[1]:


import nltk
import spacy
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb
import shap

from pathlib import Path
from warnings import simplefilter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, precision_score, recall_score, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy import sparse
from os.path import isfile
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# In[2]:


# Versões dos pacotes usados neste jupyter notebook
get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Herikc Brecher" --iversions')


# ## 2.1 Ambiente

# In[3]:


simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme()


# In[4]:


seed_ = 194
np.random.seed(seed_)


# # 3. Carregamento dos Dados

# In[5]:


# Carregando datatable de treino com as variantes
variant = pd.read_csv('data/training_variants')


# In[6]:


# Carregando datatable de treino com os textos de caso
text_data = pd.read_csv('data/training_text', sep = '\|\|', engine = 'python', names = ['ID', 'Text'], skiprows = 1)


# In[7]:


variant.head()


# In[8]:


text_data.head()


# In[9]:


# Unificando os dois datatable pela coluna 'ID' 
train_data = pd.merge(variant, text_data, on = 'ID', how = 'left')


# In[10]:


train_data.head()


# # 4. Analise Exploratoria

# In[11]:


train_data.describe(include = 'all')


# In[12]:


# Possuimos 3321 observações para treino
train_data.shape


# In[13]:


# Verificando se os tipos das colunas estão corretos
train_data.dtypes


# É analisado que possuimos valores missing para a coluna 'Text' para uma melhor generalização, e não descartar os dados iremos fazer que os dados missing sejam preenchidos pelo valor de 'Gene' + 'Variation'. 

# In[14]:


# Verificando valores missing
print(train_data.isna().sum())


# É interessante notar que possuimos 264 'Gene' diferentes e 2996 'Variation', também possuimos 1920 'Text'. Considerando que temos 3321 observações, o numero de 'Variation' unicas é muito alto, podendo prejudicar o modelo.

# In[15]:


# Verificando valores unicos
print(train_data.nunique())


# In[16]:


# Verificando valores duplicados
print(sum(train_data.duplicated()))


# ## 4.1 Analise Univariavel

# Algumas classes como 3, 9 e 8 possuimos muitos poucos dados, o que pode gerar um problema no aprendizado do nosso modelo. Induzindo a determinados víes. Um balanceamento pode auxiliar.

# In[17]:


# Criando vetor de cores
colors = ['r', 'g', 'b', 'y', 'k']

train_data['Class'].value_counts().plot(kind = 'bar', color = colors)


# In[18]:


# Funções para analise univariavel

def distribuicao_column(data, column, distribuicao_nao_cumulativa = True, distribuicao_cumulativa = True):
    # Calculando distribuição da coluna
    valores = data[column].value_counts()
    distribuicao_valores = valores / sum(valores.values)
    
    if distribuicao_nao_cumulativa:
        plt.plot(distribuicao_valores)
        plt.xlabel(column)
        plt.ylabel('Taxa de Observações')
        plt.show()
        
    if distribuicao_cumulativa:
        plt.plot(np.cumsum(distribuicao_valores))
        plt.xlabel(column)
        plt.ylabel('Taxa de Observações')
        plt.show()
        
def relevancia_classe(data, column, column_target, top = 10):
    top_significancia = round( ( sum(data.groupby(by = column).count().sort_values(by = 'ID',                                                ascending = False).head(top)[column_target]) / data.shape[0] ) * 100, 2)

    representacao_column = round( (top / data.nunique()[column]) * 100, 2)

    print(representacao_column, '% da coluna', column,'representa', top_significancia, '% da coluna', column_target)
    
def top_frequencias(data, column, column_target, ID, top = 10, colors = ['r', 'g', 'b', 'y', 'k']):
    data.groupby(by = column).count().sort_values(by = ID, ascending = False).head(top).plot(                                        kind = 'bar', ylabel = 'Frequencia', xlabel = column, y = column_target,                                        color = colors)


# Para melhor entendimento dos dados é interessante analisar o acumulo do numero de genes ao longo da sua distribuição, de forma não cumulativa e cumulativa. Já no segundo grafico mais uma vez é perceptivel que o grafico tende a crescer mais rapidamente no inicio, tendendo a uma forma exponencial.

# No primeiro grafico é perceptivel que muitos 'Genes' concentram uma maior taxa de ocorrencia, o que tende a cair rapidamente em relação aos outros.

# In[19]:


distribuicao_column(train_data, 'Gene')


# Analisando abaixo é perceptivel que os 10 Genes com maior frequência representam 36% das classes, assim essas possuem a maior relevancia para o nosso modelo. Porém isso significa que apenas 3.8% dos nossos genes representam 36% das classes.

# In[20]:


relevancia_classe(train_data, 'Gene', 'Class')


# In[21]:


top_frequencias(train_data, 'Gene', 'Class', 'ID', colors = colors)


# Apos analisar os genes é necessário analisar a 'Variation' pois essa já foi identificada que possui mais valores unicos.

# Apenas as 10 variations de maior relevancia, ou seja 0.33% dos tipos de 'Variation' representam um total de 8.85% das classes alvo.

# In[22]:


relevancia_classe(train_data, 'Variation', 'Class')


# In[23]:


top_frequencias(train_data, 'Variation', 'Class', 'ID', colors = colors)


# É perceptivel que possuem algumas poucas 'Variations' que possuem uma relevancia muito maior que as outras, porém o restante possui uma relevância similar. Observando a variancia acumulada temos o mesmo comportamento, um pulo no inicio e após um crescimento que se torna constante.

# In[24]:


distribuicao_column(train_data, 'Variation')


# ## 4.2 Tabela de Contigencia

# In[25]:


qualitativas = ['Gene', 'Variation']


# In[26]:


def crosstab_column(data, col, target, percentage = True):
    res = pd.crosstab(data[col], data[target], margins = True)
    
    if percentage:
        res = pd.crosstab(data[col], data[target], margins = True, normalize = 'index').round(4) * 100
    
    return res


# Analisando de forma generalizada é perceptivel qque determinados 'Gene' estão amplamente concentrados em uma categoria, o mesmo serve para 'Variation' que possui uma especificidade ainda maior.
# 
# Por exemplo, o 'Gene' ABL1 estã 92% na classe 2 e o restante, 7.69%, na classe 7. Já para 'Variation' o valor '1_2009trunc' esta 100% na classe 1.

# In[27]:


for col in qualitativas:
    print(crosstab_column(train_data, col, 'Class'), end = '\n\n\n')


# ## QUI-QUADRADO

# In[28]:


def qui2(data, col, target, alpha = 0.05):
    for c in col:
        cross = pd.crosstab(data[c], data[target])
        chi2, p, dof, exp = stats.chi2_contingency(cross)
        print("Qui-quadrado entre a variavel", target, "e a variavel categorica", c, ": {:0.4}".format(chi2))
        print("Apresentando um p-value de: {:0.4}".format(p))
        
        if p < alpha:
            print('A variavel', c,'possui relação direta com a variavel',target, end = '\n\n')
        else:
            print('A variavel', c,'não possui relação direta com a variavel',target, end = '\n\n')


# Analisando o teste do qui-quadrado, as nossas variaveis apontam uma alta taxa de confiança entre a relação de ambas as variaveis qualitativas com a variavem target. 

# In[29]:


qui2(train_data, qualitativas, 'Class')


# Visto que não possuimos dados quantitativos, a analise exploratoria finaliza por aqui. Concluindo que possuimos muitos valores unicos para ambas as features qualitativas. Possuimos uma grande parcela das classes atribuidas a poucos 'Genes' e 'Variations'.
# 
# Ambas as variaveis quantitativas possuem associação com a variavel target. O nivel de confiança é alto.

# # 5. Pre Processamento

# Possuimos valores NA para a coluna 'Text', para um melhor tratamento iremos subsituir o seu valor por: 'Gene' + 'Variaton'.

# In[30]:


# Verificando os valores NA
train_data[train_data.isnull().any(axis = 1)]


# In[31]:


train_data.loc[train_data['Text'].isnull(), 'Text'] = train_data['Gene'] + train_data['Variation']


# In[32]:


def limpar_texto(text):
    # Convertendo para str
    text = str(text)
    
    # Remover caracteres non-ascii
    text = ''.join(caracter for caracter in text if ord(caracter) < 128)
    
    # Convertendo para lower case
    text = text.lower()
    
    # Removendo pontuação por expressão regular
    regex = re.compile('[' + re.escape(string.punctuation) + '\\r\\t\\n]')
    text = regex.sub(' ', str(text))
    
    # Carregando stopwords em Inglês
    english_stops = set(stopwords.words('english'))
    
    # Removendo stopwords em Inglês
    # Mantendo somente palavras que não são consideradas stopwords
    text = ' '.join(palavra for palavra in text.split() if palavra not in english_stops)
    
    # Criando a estrutura baseada em uma wordnet para lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()    
    # Aplicando Lemmatization
    text = ' '.join(wordnet_lemmatizer.lemmatize(palavra) for palavra in text.split())
    
    return text


# In[33]:


def carap_data(path, data = [], column = 'x'):
    if isfile(path):
        print('Carregando conjunto de dados...')
        data = pd.read_csv(path, sep = ',')
    else:
        print('Tratanto texto...')
        data[column] = data[column].map(limpar_texto)
        print('Salvando dados...')
        data.to_csv(path, sep = ',')
    
    return data


# Iremos executar uma série de operações para realizar a limpeza do texto e generalizar para os algoritmos:
# 
# - Garantir que todo o texto é do tipo str
# - Remover caracteres não ascii
# - Converter todo o texto para minusculo
# - Remover pontuações e simbolos especiais
# - Remover stopwords (Palavras que não trazem sentido para o texto)
# - Lemmatização do texto baseado em uma wordnet
# 
# Lemmatization:
# - Forma flexionada: organizando
# - Lema: organiza
# 
# - Forma flexionada: organizado
# - Lema: organiza

# In[34]:


get_ipython().run_cell_magic('time', '', "train_data_processado = carap_data('data/treino_processado.csv', train_data, 'Text')")


# Segue comparação abaixo entre o texto da primeira linha antes de ser processado e após ser processado.

# In[35]:


train_data['Text'][0]


# In[36]:


train_data_processado['Text'][0]


# In[37]:


# Contar palavras unicas
palavras_unicas = set()

train_data_processado['Text'].str.lower().str.split().apply(palavras_unicas.update)

print(len(palavras_unicas))


# ## 5.1 TFIDF

# Iremos calcular o TFIDF das palavras para entendermos a frequencia de ocorrencia por frase x documento. Porem iremos solicitar um df minimo para a palavra ser considerada, assim não iremos contar palavras com poucas ocorrencias, também iremos considerar que pode ser considerado: unigrama, bigrama e trigrama para uma melhor contextualização. Por ultimo iremos limitar a somente 1000 features em nosso TFIDF, assim buscando as palavras com maior impacto.

# In[38]:


def gerar_TFIDF(path, data = [], max_features = 1000, ngram_range = (1, 1), min_df = 3): 
    if isfile(path):
        print('Carregando matriz TFIDF...')
        tfidf = np.load(path, allow_pickle = False)
    else:
        print('Gerando matriz TFIDF...')
        TFIDF = TfidfVectorizer(min_df = min_df, ngram_range = ngram_range, max_features = max_features)
        tfidf = TFIDF.fit_transform(data).toarray()
        np.save(path, tfidf, allow_pickle = False)
    
    print('Matriz TFIDF carregada')
    return tfidf


# In[39]:


tfidf_train = gerar_TFIDF('data/tfidf_treino.npy', train_data_processado['Text'].values, 100000, (1, 3), 3)


# In[40]:


tfidf_train_dt = pd.DataFrame(tfidf_train, index = train_data_processado.index)


# In[41]:


tfidf_train_dt.head()


# ## 5.2 Truncated SVD

# Iremos utilizar o algoritmo de TruncatedSVD para reduzir os nossos dados a componentes, a sua vantagem em relação ao PCA é que o TruncatedSVD trabalha melhor com dados esparsos. A sua diferença é que não é realizado uma centralização dos dados. Assim irá ter melhores resultados com a nossa matriz de TFIDF.

# In[42]:


def gerar_componentes_TruncatedSVD(path, n_components, data, n_iter = 50, seed = 120): 
    if isfile(path):
        print('Carregando componentes...')
        svd_componentes = np.load(path, allow_pickle = False)
    else:
        print('Gerando componentes...')
        svd = TruncatedSVD(n_components = n_components, n_iter = n_iter, random_state = seed)
        svd_componentes = svd.fit_transform(data)
        np.save(path, svd_componentes, allow_pickle = False)
    
    print('Componentes carregados')
    return svd_componentes


# In[43]:


n_components = 1000
truncated_train = gerar_componentes_TruncatedSVD('data/matriz_esparsa_treino.npy', n_components,                                                 tfidf_train_dt, 50, seed_)


# In[44]:


truncated_train_dt = pd.DataFrame(truncated_train)


# In[45]:


truncated_train_dt.head()


# In[46]:


# Calcular taxa de variancia por componente
# Executar só se o grafico da variancia não estiver exibindo no output

svd = TruncatedSVD(n_components = 1000, n_iter = 50, random_state = seed_)
svd_componentes = svd.fit_transform(tfidf_train_dt)

variancia = svd.explained_variance_ratio_
variancia_acumulada = np.cumsum(variancia * 100)


# Para o nosso caso 1000 componentes será o suficiente para o nosso modelo cobrir os dados, então iremos de 100.000 features para 1000 componentes, assim reduzindo a nossa dimensionalidade.

# In[47]:


plt.ylabel('Variancia')
plt.xlabel('Componentes')
plt.title('Analise de PCA')
plt.ylim(10, 100)
plt.xlim(0, n_components)
plt.plot(variancia_acumulada)


# In[48]:


# Renomeando colunas
truncated_train_dt.columns = [('Componente ', i) for i in range(1, n_components + 1)]


# ## 5.3 One Hot Encoding

# Iremos utilizar one hot encoding para converter as nossas classes categoricas como 'Gene' e 'Variation' em numericas. Porém essas classes não apresentam hierarquia, assim iremos optar por One Hot Encoding e não Labelencode.

# In[49]:


# Carregando somente as colunas de 'Gene' e 'Variation'
train_data_one_hot = train_data_processado[['Gene', 'Variation']]

# Removendo as colunas de 'Gene' e 'Variation'
train_data_temp = train_data_processado.drop(['Gene', 'Variation'], axis = 1)


# In[50]:


# Aplicando OneHotEncoder
onehot = OneHotEncoder(dtype = int)

train_data_one_hot = onehot.fit_transform(train_data_one_hot)


# In[51]:


# Convertendo o resultado para dataframe
train_data_one_hot_dt = pd.DataFrame(train_data_one_hot.toarray())


# In[52]:


# Realizando join nas colunas extras
train_data_one_hot_dt = train_data_temp.join(train_data_one_hot_dt) 


# In[53]:


train_data_one_hot_dt.head()


# Agora iremos concatenar os componentes do TruncatedSVD, gerados apartir do TFIDF no datatable final.

# In[54]:


# Removendo coluna Text pois o TFIDF irá sobrepor
train_data_final = train_data_one_hot_dt.drop('Text', axis = 1)


# In[55]:


# Concatenando componentes com onehot
train_data_final = pd.concat([train_data_final, truncated_train_dt], axis = 1)


# In[56]:


train_data_final.head()


# In[57]:


train_data_final = train_data_final.drop('ID', axis = 1)


# ## 5.4 Salvando/Carregando Dataset

# Antes de iniciarmos a fase de treino é interessante salvarmos em um aruivo '.csv' os dados, assim para execuções futuras não é necessário executar as etapas anteriores.

# In[58]:


path = 'data/treino_final.csv'

if isfile(path):
    print('Carregando dataset...')
    train_data_final = pd.read_csv(path, sep = ',')
else:
    print('Salvando dataset...')
    train_data_final.to_csv(path, sep = ',')


# In[59]:


train_data_final = train_data_final.drop('Unnamed: 0', axis = 1)


# In[60]:


train_data_final.head()


# # 6 Treino, Teste e Validação

# Iremos separar os nossos dados em treino, teste e validação. Onde os dados de validação serão retirados dos dados de treino e utilizados para otimização durante o treinamento. Já os dados de teste serão separados de forma que não interfiram no resultado final.

# In[61]:


X = train_data_final.drop('Class', axis = 1)
y = train_data_final['Class'].values


# In[62]:


# Separando o conjunto principal em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.3, random_state = seed_)

# Separando o conjunto de treino em treino e validação
X_train_, X_validacao, y_train_, y_validacao = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.2)

# Balanceando o conjunto de treino original
oversample = SMOTE(random_state = seed_)
X_train_resample, y_train_resample = oversample.fit_resample(X_train, y_train)

# Balanceando o conjunto de treino que foi separado em validacao
oversample = SMOTE(random_state = seed_)
X_train_resample_, y_train_resample_ = oversample.fit_resample(X_train_, y_train_)


# In[63]:


print('Observaçoes em treino:', X_train_.shape[0])
print('Observaçoes em treino balanceado:', X_train_resample.shape[0])
print('Observaçoes em teste:', X_test.shape[0])
print('Observaçoes em validação:', X_validacao.shape[0])


# In[64]:


# Convertendo dataframe para matriz esparsa
X_test_original = X_test.copy()
X_train = sparse.csr_matrix(X_train.values)
X_train_ = sparse.csr_matrix(X_train_.values)
X_train_resample = sparse.csr_matrix(X_train_resample.values)
X_train_resample_ = sparse.csr_matrix(X_train_resample_.values)
X_test = sparse.csr_matrix(X_test.values)
X_validacao = sparse.csr_matrix(X_validacao.values)


# ## 6.1 Distribuição por conjunto

# In[65]:


def distribuicao(data, colors = ['r', 'g', 'b', 'y', 'k'], verbose = False):
    data2 = data.value_counts().sort_index()
    data2.plot(kind = 'bar', color = colors, stacked = True)
    plt.xlabel('Class')
    plt.ylabel('Ocorrencias')
    plt.show()
      
    if verbose:
        sorted_class = np.argsort(-data2.values)
        for i in sorted_class:
            print('Observações na classe', i + 1, ':',                  data2.values[i], '(',                  np.round((data2.values[i]/data.shape[0]*100), 3), '%)')


# In[66]:


distribuicao(pd.DataFrame(y_test), verbose = True)


# In[67]:


distribuicao(pd.DataFrame(y_train), verbose = True)


# In[68]:


distribuicao(pd.DataFrame(y_train_resample), verbose = True)


# In[69]:


distribuicao(pd.DataFrame(y_validacao), verbose = True)


# # 7. Modelagem Preditiva

# ## 7.1 Criação de Modelos Bases

# Primeiramente iremos criar os modelos bases para entender qual comportamento é melhor para os nossos modelos preditivos. Os modelos bases irão ser criados com as seguintes configurações:
# 
#     - Balanceamento em treino = True or False
#     - Calibracao = True or False
#     - Conjunto de calibracao: Treino ou validacao
#    
# Para as configuracoes acima, iremos utilizar os seguintes algoritmos: Regressão Logistica, Linear SVM, Random Forest, XGBoost e KNN.

# In[70]:


configuracoes = []


# In[71]:


def executaModelo(modelo, treino, teste, validacao, calibration = False):
    try:
        # Treina o modelo
        modelo.fit(treino[0], treino[1])

        if calibration:
            # Instancia a calibração
            calibration = CalibratedClassifierCV(base_estimator = modelo, method = 'sigmoid', cv = 3)
            
            # Aplica a calibração
            calibration.fit(validacao[0], validacao[1])

            # Realiza as previsões
            pred = calibration.predict_proba(teste[0])
        else:
            # Realiza as previsões de acordo com o tipo do modelo probabilistico ou não
            try:           
                pred = modelo.predict_proba(teste[0])
            except:
                pred = modelo.predict(teste[0])

        # Calcula a loss
        loss = log_loss(teste[1], pred)
        
        return loss    
    except:
        print('Treino ignorado')
    
    return -1


# In[72]:


def executaModelos(modelo, data_treino_balanceado, data_treino_desbalanceado, data_validacao,                  data_treino_validacao_balanceado, data_treino_validacao_desbalanceado, data_teste, algoritmo):
    global configuracoes
    
    # Balanceamento em treino, calibracao, conjunto de calibracao em treino
    print('Iniciando treino 1...')
    
    loss = executaModelo(modelo, data_treino_balanceado, data_teste, data_treino_balanceado, True)
    
    if loss != -1:
        configuracoes.append({'Algoritmo': algoritmo, 'Balanceamento': True,                              'Calibracao': True, 'Conjunto Calibracao': 'Treino', 'Loss': loss})
    
    # Balanceamento em treino, calibracao, conjunto de calibracao em validacao
    print('Iniciando treino 2...')
    
    loss = executaModelo(modelo, data_treino_validacao_balanceado, data_teste, data_validacao, True)
    
    if loss != -1:
        configuracoes.append({'Algoritmo': algoritmo, 'Balanceamento': True,                            'Calibracao': True, 'Conjunto Calibracao': 'Validacao', 'Loss': loss})
    
    # Balanceamento em treino, sem calibracao
    print('Iniciando treino 3...')
    
    loss = executaModelo(modelo, data_treino_balanceado, data_teste, [], False)
    
    if loss != -1:
        configuracoes.append({'Algoritmo': algoritmo, 'Balanceamento': True,                              'Calibracao': False, 'Conjunto Calibracao': None, 'Loss': loss})
    
    # Desbalanceamento em treino, calibracao, conjunto de calibracao em treino
    print('Iniciando treino 4...')
    
    loss = executaModelo(modelo, data_treino_desbalanceado, data_teste, data_treino_desbalanceado, True)
    
    if loss != -1:
        configuracoes.append({'Algoritmo': algoritmo, 'Balanceamento': False,                            'Calibracao': True, 'Conjunto Calibracao': 'Treino', 'Loss': loss})
    
    # Desbalanceamento em treino, calibracao, conjunto de calibracao em validacao
    print('Iniciando treino 5...')
    
    loss = executaModelo(modelo, data_treino_validacao_desbalanceado, data_teste, data_validacao, True)
    
    if loss != -1:
        configuracoes.append({'Algoritmo': algoritmo, 'Balanceamento': False,                              'Calibracao': True, 'Conjunto Calibracao': 'Validacao', 'Loss': loss})
    
    # Desbalanceamento em treino, sem calibracao
    print('Iniciando treino 6...')
    
    loss = executaModelo(modelo, data_treino_desbalanceado, data_teste, [], False)
    
    if loss != -1:
        configuracoes.append({'Algoritmo': algoritmo, 'Balanceamento': False,                              'Calibracao': False, 'Conjunto Calibracao': None, 'Loss': loss})
        


# In[73]:


# Executando todas as configurações citadas para o algoritmo de Regressão Logistica
executaModelos(SGDClassifier(loss = 'log', class_weight = 'balanced', random_state = seed_),             [X_train_resample, y_train_resample], [X_train, y_train], [X_validacao, y_validacao],             [X_train_resample_, y_train_resample_], [X_train_, y_train_], [X_test, y_test], 'Regressão Logistica')


# In[74]:


# Executando todas as configurações citadas para o algoritmo de Linear SVM
executaModelos(SGDClassifier(loss = 'hinge', class_weight = 'balanced', random_state = seed_),             [X_train_resample, y_train_resample], [X_train, y_train], [X_validacao, y_validacao],             [X_train_resample_, y_train_resample_], [X_train_, y_train_], [X_test, y_test], ' Linear SVM')


# In[75]:


# Executando todas as configurações citadas para o algoritmo de KNN
executaModelos(KNeighborsClassifier(),             [X_train_resample, y_train_resample], [X_train, y_train], [X_validacao, y_validacao],             [X_train_resample_, y_train_resample_], [X_train_, y_train_], [X_test, y_test], 'KNN')


# In[76]:


# Executando todas as configurações citadas para o algoritmo de Random Forest
executaModelos(RandomForestClassifier(random_state = seed_),             [X_train_resample, y_train_resample], [X_train, y_train], [X_validacao, y_validacao],             [X_train_resample_, y_train_resample_], [X_train_, y_train_], [X_test, y_test], 'Random Forest')


# In[77]:


# Ordena os modelos de acordo com a menor Loss
sorted_configuracoes = sorted(configuracoes, key = lambda k: k['Loss'])


# Abaixo visualizamos que os 2 melhores modelos são do mesmo algoritmo, Random Forest. A unica diferença entre eles é em relação ao balanceamento um possuo e o outro não. 
# 
# Configuração do melhor modelo:
#   - 'Algoritmo': 'Random Forest',
#   - 'Balanceamento': False,
#   - 'Calibracao': True,
#   - 'Conjunto Calibracao': 'Treino',
#   - 'Loss': 1.21

# In[78]:


sorted_configuracoes_dt = pd.DataFrame(sorted_configuracoes)


# In[79]:


sorted_configuracoes_dt.head()


# In[80]:


def save_model(modelo):
    shortFileName = '000'
    fileName = 'models/0001.model'
    fileObj = Path(fileName)

    index = 1
    while fileObj.exists():
        index += 1
        fileName = 'models/' + shortFileName + str(index) + '.model'
        fileObj = Path(fileName)

    # Salvar modelo
    pickle.dump(modelo, open(fileName, 'wb'))
    
    return fileName

def plot_general_report(modelo, y_true, y_pred, save = False):
    # Calculando Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculando Precision Matrix
    precision = (cm/cm.sum(axis=0))
    
    # Calculando Recall Matrix
    recall = (((cm.T)/(cm.sum(axis=1))).T)
    
    labels = range(1, 10)
    
    # Plot da Confusion Matrix
    print("-"*20, "Confusion matrix", "-"*20)
    plt.figure(figsize = (20,7))
    sns.heatmap(cm, annot = True, cmap = "YlGnBu", fmt = ".3f", xticklabels = labels, yticklabels = labels)
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.show()

    # Plot da Precision Matrix
    print("-"*20, "Precision matrix", "-"*20)
    plt.figure(figsize = (20,7))
    sns.heatmap(precision, annot = True, cmap = "YlGnBu", fmt = ".3f", xticklabels = labels, yticklabels = labels)
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.show()
    
    # Plot da Recall Matrix
    print("-"*20, "Recall matrix", "-"*20)
    plt.figure(figsize = (20,7))
    sns.heatmap(recall, annot = True, cmap = "YlGnBu", fmt = ".3f", xticklabels = labels, yticklabels = labels)
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.show()
    
    # Relatorio Macro/Micro
    recall = round( recall_score(y_true, y_pred, average = 'macro', zero_division = 0), 4)
    precision = round( precision_score(y_true, y_pred, average = 'macro', zero_division = 0), 4)
    f1_score_ = round( f1_score(y_true, y_pred, average = 'macro', zero_division = 0), 4)
    
    print('Macro Precision:', precision)
    print('Macro Recall:', recall)
    print('F1-Score:', f1_score_)
    
    # Salvando modelo sem sobreescrever arquivos existentes
    if save:
        fileName = save_model(modelo)
        
        return fileName


# In[81]:


# Modelo
modelo = RandomForestClassifier(random_state = seed_)
modelo.fit(X_train, y_train)

# Calibração
calibration = CalibratedClassifierCV(base_estimator = modelo, method = 'sigmoid', cv = 3)
calibration.fit(X_train, y_train)

# Realiza as previsões
pred = calibration.predict(X_test)
pred_prob = calibration.predict_proba(X_test)

# Calcula a loss
loss = log_loss(y_test, pred_prob)

print('Loss:', round(loss, 4))


# In[82]:


plot_general_report(calibration, y_test, pred, save = True)


# ## 7.2 Tuning Random Forest

# Com a escolha do melhor algoritmo como sendo Random Forest, iremos realizar o seu Tuning. Tentar encontrar os melhores hiperparametros.

# In[83]:


def treina_GridSearchCV(modelo, params_, x_treino, y_treino, x_teste, y_teste,                        n_jobs = 20, cv = 5, refit = True, scoring = None, salvar_resultados = False,                       report_treino = False):
    grid = GridSearchCV(modelo, params_, n_jobs = n_jobs, cv = cv, refit = refit, scoring = scoring)
    
    print('Iniciando Treino...')
    grid.fit(x_treino, y_treino)
    print('Treino finalizado')
    
    print('Realizando predições')
    pred = grid.predict(x_teste)
    modelo_ = grid.best_estimator_
    print('Finalizando predições')
    
    print(grid.best_params_)
    
    target_names = range(1, 10)
    
    print('-'*20, 'Report Para Dados de Teste', '-'*20)
    
    plot_general_report(modelo, y_test, pred, save = True)
    
    if report_treino:
        print('-'*20, 'Report Para Dados de Treino', '-'*20)
        pred_treino = grid.predict(x_treino)
        
        plot_general_report(modelo, y_treino, pred_treino, save = False)
    
    if salvar_resultados:
        resultados_df = pd.DataFrame(grid.cv_results_)
        
        return resultados_df 


# Verificando as métricas do modelo inicial tanto em treino e teste, é perceptivel que o nosso modelo sofre de overfitting, esse pode ser um dos problemas para uma baixa precisão em dados de teste. Assim iremos ter que podar a nossa arvore para atingir melhores resultados.

# In[84]:


get_ipython().run_cell_magic('time', '', "\n# Comparativo modelo base v1\nparams = {\n    'random_state': [seed_]\n}\n\nresultados = treina_GridSearchCV(RandomForestClassifier(), params, X_train, y_train, X_test, y_test, cv = 3,\\\n                    report_treino = True, salvar_resultados = True)")


# In[85]:


# Modelo v2
params = {
    'n_estimators': [250, 500, 1000],
    'criterion': ['gini', 'entropy']
}

modelo = RandomForestClassifier(random_state = seed_, class_weight = 'balanced', n_jobs = 4)

resultados = treina_GridSearchCV(modelo, params, X_train, y_train, X_test, y_test, cv = 3,                    report_treino = True, salvar_resultados = True)


# In[86]:


# Modelo v3
params = {
    'n_estimators': [500, 1000],
    'criterion': ['gini'],
    'max_depth': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 3]
}

modelo = RandomForestClassifier(random_state = seed_, class_weight = 'balanced', n_jobs = 4)

resultados = treina_GridSearchCV(modelo, params, X_train, y_train, X_test, y_test, cv = 3,                    report_treino = True, salvar_resultados = True)


# In[87]:


# Modelo v4
params = {
    'n_estimators': [500, 1000],
    'criterion': ['gini'],
    'max_depth': [4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4, 6]
}

modelo = RandomForestClassifier(random_state = seed_, class_weight = 'balanced', n_jobs = 4)

resultados = treina_GridSearchCV(modelo, params, X_train, y_train, X_test, y_test, cv = 3,                    report_treino = True, salvar_resultados = True)


# In[88]:


# Modelo v5
params = {
    'n_estimators': [500, 1000],
    'criterion': ['gini'],
    'max_depth': [6, 8, 10],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [4, 6, 8],
    'max_features': ['sqrt', 'log2']
}

modelo = RandomForestClassifier(random_state = seed_, class_weight = 'balanced', n_jobs = 4)

resultados = treina_GridSearchCV(modelo, params, X_train, y_train, X_test, y_test, cv = 3,                    report_treino = True, salvar_resultados = True)


# In[89]:


# Modelo v6
params = {
    'n_estimators': [500, 1000],
    'criterion': ['gini'],
    'max_depth': [6, 8, 10],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [4, 6, 8],
    'max_features': ['sqrt', 'log2'],
    'max_leaf_nodes': [2, 4, 6, 8]
}

modelo = RandomForestClassifier(random_state = seed_, class_weight = 'balanced', n_jobs = 4)

resultados = treina_GridSearchCV(modelo, params, X_train, y_train, X_test, y_test, cv = 3,                    report_treino = True, salvar_resultados = True)


# In[90]:


# Modelo v7
modelo = RandomForestClassifier(random_state = seed_, class_weight = 'balanced', n_jobs = 4,                               criterion = 'gini', max_depth = 8, max_features = 'sqrt',                                min_samples_leaf = 4, min_samples_split = 6, n_estimators = 1000)
    
modelo.fit(X_train, y_train)
pred = modelo.predict(X_test)

print('-'*20, 'Report Para Dados de Teste', '-'*20)
    
plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(X_train)

plot_general_report(modelo, y_train, pred_treino, save = False)


# In[91]:


# Modelo v8 - SMOTE em Treino
modelo = RandomForestClassifier(random_state = seed_, class_weight = 'balanced', n_jobs = 4, criterion = 'gini',                                max_depth = 8, max_features = 'auto',                                min_samples_leaf = 4, min_samples_split = 6, n_estimators = 1000)
    
modelo.fit(X_train_resample, y_train_resample)
pred = modelo.predict(X_test)

print('-'*20, 'Report Para Dados de Teste', '-'*20)
    
plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(X_train_resample)

plot_general_report(modelo, y_train_resample, pred_treino, save = False)


# In[92]:


# Modelo v9 - SMOTE em Treino ----- FINAL -----
modelo = RandomForestClassifier(random_state = seed_, class_weight = 'balanced', n_jobs = 4, criterion = 'entropy',                                max_depth = 8, max_features = 'auto',                                min_samples_leaf = 4, min_samples_split = 6, n_estimators = 1000)
    
modelo.fit(X_train_resample, y_train_resample)
pred = modelo.predict(X_test)
pred_prob = modelo.predict_proba(X_test)

# Calcula a loss
loss = log_loss(y_test, pred_prob)

print('-'*20, 'Report Para Dados de Teste', '-'*20)
print('Loss:', round(loss, 4))

plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(X_train_resample)

plot_general_report(modelo, y_train_resample, pred_treino, save = False)


# # 8 Conclusão sobre Random Forest

# Através da execução de diversos algoritmos e tuning foi observado que conforme o Log Loss aumentava conseguiamos uma maior precisão em algumas classes. Porém isso significa que estamos aumentando nossa precisão ao mesmo tempo que aumentamos a nossa taxa de incerteza sobre as previsões.
# 
# Assim sendo, foi optado por aumentar a precisão ao inves de diminuir o Log Loss. Um dos problemas observados ao longo do processo foi o overfitting que apesar de alto conseguimos reduzir um pouco.
# 
# Foi realizado testes usando dados balanceados e desbalanceados, com ou sem calibração, conjunto de validação... Foi alterado os parametros durante o pre processamento, aumentando e diminuindo o total de componentes. Também foram realizados testes com diferentes numeros de 'features' geradas pelo procedo TFIDF.
# 
# Sugestões de melhoria:
# 
#     - Otimizar outros algoritmos como XGBoost e/ou SVM.
#     
# Hiper parametrização do melhor modelo:
# 
#     - Algoritmo: Random Forest
#        - random_state = 194
#        - class_weight = 'balanced'
#        - n_jobs = 4
#        - criterion = 'entropy'
#        - max_depth = 8
#        - max_features = 'auto'
#        - min_samples_leaf = 4
#        - min_samples_split = 6
#        - n_estimators = 1000

# ## 8.1 Execução do Melhor Modelo

# In[93]:


#----- FINAL -----
modelo = RandomForestClassifier(random_state = seed_, class_weight = 'balanced', n_jobs = 4, criterion = 'entropy',                                max_depth = 8, max_features = 'auto',                                min_samples_leaf = 4, min_samples_split = 6, n_estimators = 1000)
    
modelo.fit(X_train_resample, y_train_resample)
pred = modelo.predict(X_test)
pred_prob = modelo.predict_proba(X_test)

# Calcula a loss
loss = log_loss(y_test, pred_prob)

print('Loss:', round(loss, 4))

plot_general_report(modelo, y_test, pred, save = True)


# ## 8.2 Analise do Melhor Modelo Random Forest

# A analise do modelo realizada abaixa comprova que os componentes gerados pelo apartir do TFIDF são de extrema importância. Assim podemos tentar aumentar a qualidade dos nossos componentes ou da extração do TFIDF. O grande problema é em relação ao custo computacional gerado com o aumento dos anteriores.

# In[94]:


# Criando o explainer do modelo
explainer = shap.TreeExplainer(modelo)


# In[95]:


# Interpretação da predição 0
shap.initjs()
shap_data = X_test_original.iloc[0]
shap_values = explainer.shap_values(shap_data)
shap.force_plot(explainer.expected_value[1], shap_values[1], shap_data)


# In[96]:


# Interpretação da predição 0 a 5
# Executar e após visualizar limpar a celula. Fica muito pesado no notebook
'''shap.initjs()
shap_data = X_test_original.iloc[0:5]
shap_values = explainer.shap_values(shap_data)
shap.force_plot(explainer.expected_value[1], shap_values[1], shap_data)'''


# In[97]:


# Interpretação do modelo em relação as 10 primeiras predições
shap.initjs()
shap_data = X_test_original.iloc[0:10]
shap_values = explainer.shap_values(shap_data)
shap.summary_plot(shap_values[1], shap_data)


# # 9. Treinamento XGBoost

# Após verificar que o algoritmo RandomForest estava tendendo muito ao overfitting e com dificuldade de diminui-lo por hiperparametro. Foi optado por utilizar o XGBoost que já apresentou resultados melhores para casos similares.  

# In[98]:


# Modelo v1

dtrain = xgb.DMatrix(data = X_train_resample, label = y_train_resample)
dtest = xgb.DMatrix(data = X_test, label = y_test)

params = {
    'objective': 'multi:softmax',
    'num_class': 10,
    'random_state': seed_
}

modelo = xgb.train(params = params, dtrain = dtrain)
    
pred = modelo.predict(dtest)

print('-'*20, 'Report Para Dados de Teste', '-'*20)

plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(dtrain)

plot_general_report(modelo, y_train_resample, pred_treino, save = False)


# In[99]:


# Modelo v2

dtrain = xgb.DMatrix(data = X_train_resample, label = y_train_resample)
dtest = xgb.DMatrix(data = X_test, label = y_test)

params = {
    # Definições de ambiente de treino
    'objective': 'multi:softmax',
    'num_class': 10,
    'random_state': seed_,
    'nthread': 2,
    # Hiperparametros a serem ajustados
    'colsample_bynode': 1, 
    'colsample_bytree': 1,
    'gamma': 0,
    'learning_rate': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,  
    'subsample': 0.8,
    'colsample_bylevel': 1
}

modelo = xgb.train(params = params, dtrain = dtrain)
    
pred = modelo.predict(dtest)

print('-'*20, 'Report Para Dados de Teste', '-'*20)

plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(dtrain)

plot_general_report(modelo, y_train_resample, pred_treino, save = False)


# In[100]:


# Modelo v3

dtrain = xgb.DMatrix(data = X_train_resample, label = y_train_resample)
dtest = xgb.DMatrix(data = X_test, label = y_test)

params = {
    # Definições de ambiente de treino
    'objective': 'multi:softmax',
    'num_class': 10,
    'random_state': seed_,
    'nthread': 2,
    # Hiperparametros a serem ajustados
    'colsample_bynode': 1, 
    'colsample_bytree': 1,
    'gamma': 0,
    'learning_rate': 0.5,
    'max_depth': 6,
    'min_child_weight': 1,  
    'subsample': 0.8,
    'colsample_bylevel': 1
}

modelo = xgb.train(params = params, dtrain = dtrain)
    
pred = modelo.predict(dtest)

print('-'*20, 'Report Para Dados de Teste', '-'*20)

plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(dtrain)

plot_general_report(modelo, y_train_resample, pred_treino, save = False)


# In[101]:


# Modelo v4

dtrain = xgb.DMatrix(data = X_train_resample, label = y_train_resample)
dtest = xgb.DMatrix(data = X_test, label = y_test)

params = {
    # Definições de ambiente de treino
    'objective': 'multi:softmax',
    'num_class': 10,
    'random_state': seed_,
    'nthread': 2,
    # Hiperparametros a serem ajustados
    'colsample_bynode': 1, 
    'colsample_bytree': 1,
    'gamma': 0.5,
    'learning_rate': 0.5,
    'max_depth': 6,
    'min_child_weight': 1,  
    'subsample': 0.8,
    'colsample_bylevel': 1
}

modelo = xgb.train(params = params, dtrain = dtrain)
    
pred = modelo.predict(dtest)

print('-'*20, 'Report Para Dados de Teste', '-'*20)

plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(dtrain)

plot_general_report(modelo, y_train_resample, pred_treino, save = False)


# In[102]:


# Modelo v5

dtrain = xgb.DMatrix(data = X_train_resample, label = y_train_resample)
dtest = xgb.DMatrix(data = X_test, label = y_test)

params = {
    # Definições de ambiente de treino
    'objective': 'multi:softmax',
    'num_class': 10,
    'random_state': seed_,
    'nthread': 2,
    # Hiperparametros a serem ajustados
    'colsample_bynode': 1, 
    'colsample_bytree': 1,
    'gamma': 0.5,
    'learning_rate': 0.5,
    'max_depth': 10,
    'min_child_weight': 1,  
    'subsample': 0.8,
    'colsample_bylevel': 1
}

modelo = xgb.train(params = params, dtrain = dtrain)
    
pred = modelo.predict(dtest)

print('-'*20, 'Report Para Dados de Teste', '-'*20)

plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(dtrain)

plot_general_report(modelo, y_train_resample, pred_treino, save = False)


# In[103]:


# Modelo v6

dtrain = xgb.DMatrix(data = X_train_resample, label = y_train_resample)
dtest = xgb.DMatrix(data = X_test, label = y_test)

params = {
    # Definições de ambiente de treino
    'objective': 'multi:softmax',
    'num_class': 10,
    'random_state': seed_,
    'nthread': 2,
    # Hiperparametros a serem ajustados
    'colsample_bynode': 1, 
    'colsample_bytree': 1,
    'gamma': 0.5,
    'learning_rate': 0.5,
    'max_depth': 8,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bylevel': 1
}

modelo = xgb.train(params = params, dtrain = dtrain)
    
pred = modelo.predict(dtest)

print('-'*20, 'Report Para Dados de Teste', '-'*20)

plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(dtrain)

plot_general_report(modelo, y_train_resample, pred_treino, save = False)


# # 10 Conclusão sobre XGBoost

# Através do treino do algoritmo de XGBoost. Não foi possivel aumentar a precisçao do modelo, isso se da pelo sua alta complexidade, talvez seja melhor optar por modelos como SVM que podem lidar melhor com altas dimensionaliades. 
# 
# Sugestões de melhoria:
# 
#     - Otimizar outros algoritmos como Linear SVM e Regressão Logistica.
#     
# Hiper parametrização do melhor modelo:
# 
#     - Algoritmo: XGBoost
#             - 'objective': 'multi:softmax'
#             - 'num_class': 10
#             - 'random_state': 194
#             - 'nthread': 2
#             - 'colsample_bynode': 1
#             - 'colsample_bytree': 1
#             - 'gamma': 0.5
#             - 'learning_rate': 0.5
#             - 'max_depth': 6
#             - 'min_child_weight': 1
#             - 'subsample': 0.8
#             - 'colsample_bylevel': 1

# ## 10.1 Executando Melhor Modelo XGBoost

# In[104]:


dtrain = xgb.DMatrix(data = X_train_resample, label = y_train_resample)
dtest = xgb.DMatrix(data = X_test, label = y_test)

params = {
    # Definições de ambiente de treino
    'objective': 'multi:softmax',
    'num_class': 10,
    'random_state': seed_,
    'nthread': 2,
    # Hiperparametros a serem ajustados
    'colsample_bynode': 1, 
    'colsample_bytree': 1,
    'gamma': 0.5,
    'learning_rate': 0.5,
    'max_depth': 6,
    'min_child_weight': 1,  
    'subsample': 0.8,
    'colsample_bylevel': 1
}

modelo = xgb.train(params = params, dtrain = dtrain)
    
pred = modelo.predict(dtest)

print('-'*20, 'Report Para Dados de Teste', '-'*20)

plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(dtrain)

plot_general_report(modelo, y_train_resample, pred_treino, save = False)


# ## 10.2 Analise do Melhor Modelo XGBoost

# O XGboost possui um comportamento diferente, onde as variaveis geradas pelo One Hot Enconde de 'Gene' e 'Variation' possuem um destaque um pouco maior. Porém o seu desempenho ainda não foi superior ao Random Forest.

# In[105]:


# Criando o explainer do modelo
explainer = shap.TreeExplainer(modelo)


# In[106]:


# Interpretação da predição 0
shap.initjs()
shap_data = X[0:1]
shap_values = explainer.shap_values(shap_data)
shap.force_plot(explainer.expected_value[1], shap_values[1], shap_data)


# In[107]:


# Interpretação da predição 0 a 5
# Executar e após visualizar limpar a celula. Fica muito pesado no notebook
'''
shap.initjs()
shap_data = X[0:5]
shap_values = explainer.shap_values(shap_data)
shap.force_plot(explainer.expected_value[1], shap_values[1], shap_data)
'''


# In[108]:


# Interpretação do modelo em relação as 10 primeiras predições
shap.initjs()
shap_data = X[0:10]
shap_values = explainer.shap_values(shap_data)
shap.summary_plot(shap_values[1], shap_data)


# # 11 Treinamento Regressão Logistica

# Após testar dois modelos baseado em Arvores, percebemos que esses não se adaptam tão bem aos dados que possuimos. Com isso iremos tentar utilizar a Regressão Logistica. Visto que esse tende a se adaptar melhor a generalizações massivas, com probabilidades como TFIDF.

# In[109]:


# Modelo Base v1
# Dados balanceados

modelo = SGDClassifier(loss = 'log', class_weight = 'balanced', random_state = seed_)

modelo.fit(X_train_resample, y_train_resample)

pred_proba = modelo.predict_proba(X_test)
pred = modelo.predict(X_test)

print('-'*20, 'Report Para Dados de Teste', '-'*20)

print('Log loss: ', log_loss(y_test, pred_proba))

plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(X_train_resample)

plot_general_report(modelo, y_train_resample, pred_treino, save = False)


# In[110]:


# Modelo v2
# Dados balanceados

# Hiperparametros
alpha = [10 ** x for x in range(-8, 5)]

for alpha_ in alpha:
    modelo = SGDClassifier(loss = 'log', class_weight = 'balanced', random_state = seed_, alpha = alpha_, n_jobs = -1)

    modelo.fit(X_train_resample, y_train_resample)

    pred_proba = modelo.predict_proba(X_test)

    print('Alpha =', alpha_, 'Log loss: ', log_loss(y_test, pred_proba))


# In[111]:


# Modelo v3
# Dados balanceados

modelo = SGDClassifier(loss = 'log', class_weight = 'balanced', random_state = seed_, alpha = 0.0001, n_jobs = -1,                      early_stopping = True, validation_fraction = 0.2)

modelo.fit(X_train_resample, y_train_resample)

pred_proba = modelo.predict_proba(X_test)

print('Log loss: ', log_loss(y_test, pred_proba))


# In[112]:


# Modelo v4
# Dados balanceados

modelo = SGDClassifier(loss = 'log', class_weight = 'balanced', random_state = seed_, alpha = 0.0001, n_jobs = -1,                      early_stopping = True, validation_fraction = 0.1)

modelo.fit(X_train_resample, y_train_resample)

pred_proba = modelo.predict_proba(X_test)

print('Log loss: ', log_loss(y_test, pred_proba))


# In[113]:


# Modelo v5
# Dados desbalanceados

# Hiperparametros
alpha = [10 ** x for x in range(-8, 5)]

for alpha_ in alpha:
    modelo = SGDClassifier(loss = 'log', class_weight = 'balanced', random_state = seed_, alpha = alpha_, n_jobs = -1)

    modelo.fit(X_train, y_train)

    pred_proba = modelo.predict_proba(X_test)

    print('Alpha =', alpha_, 'Log loss: ', log_loss(y_test, pred_proba))


# In[114]:


# Modelo v6
# Dados desbalanceados

modelo = SGDClassifier(loss = 'log', class_weight = 'balanced', random_state = seed_, alpha = 0.0001, n_jobs = -1,                      early_stopping = True, validation_fraction = 0.1)

modelo.fit(X_train, y_train)

pred_proba = modelo.predict_proba(X_test)

print('Log loss: ', log_loss(y_test, pred_proba))


# In[115]:


# Analise modelo v6

pred = modelo.predict(X_test)

print('-'*20, 'Report Para Dados de Teste', '-'*20)

plot_general_report(modelo, y_test, pred, save = True)

print('-'*20, 'Report Para Dados de Treino', '-'*20)
pred_treino = modelo.predict(X_train_resample)

plot_general_report(modelo, y_train_resample, pred_treino, save = False)


# In[116]:


# Modelo v7
# Dados desbalanceados

modelo = SGDClassifier(loss = 'log', class_weight = 'balanced', random_state = seed_, alpha = 0.0001, n_jobs = -1,                      early_stopping = True, validation_fraction = 0.2)

modelo.fit(X_train, y_train)

pred_proba = modelo.predict_proba(X_test)

print('Log loss: ', log_loss(y_test, pred_proba))


# ## 11.1 Executando Melhor Modelo Regressão Logistica

# In[117]:


# Modelo Final

modelo = SGDClassifier(loss = 'log', class_weight = 'balanced', random_state = seed_, alpha = 0.0001, n_jobs = -1)

modelo.fit(X_train, y_train)

pred_proba = modelo.predict_proba(X_test)

print('Log loss: ', log_loss(y_test, pred_proba))


# ## 11.2 Analise Melhor Modelo Regressão Logistica

# A Regressão Logistica conseguiu apresentar o menor log loss, o que é positivo. Ainda em relação aos outros algoritmos foi perceptivel que a Regressão Logistica deu mais valor a variaveis geradas pelo One Hot Encoder. Comportamento esse que já tinhamos analisado no XGBoost.

# In[118]:


# Criando o explainer do modelo
explainer = shap.LinearExplainer(modelo, X_train)


# In[119]:


# Interpretação da predição 0
shap.initjs()
shap_data = X[0:1]
shap_values = explainer.shap_values(shap_data)
shap.force_plot(explainer.expected_value[1], shap_values[1], shap_data)


# In[120]:


# Interpretação da predição 0 a 5
# Executar e após visualizar limpar a celula. Fica muito pesado no notebook
'''
shap.initjs()
shap_data = X[0:5]
shap_values = explainer.shap_values(shap_data)
shap.force_plot(explainer.expected_value[1], shap_values[1], shap_data)
'''


# In[121]:


# Interpretação do modelo em relação as 10 primeiras predições
shap.initjs()
shap_data = X[0:10]
shap_values = explainer.shap_values(shap_data)
shap.summary_plot(shap_values[1], shap_data)


# # 12 Conclusão

# Após um trabalho grande na analise exploratoria, pre-processamento, modelagem... Foi possivel chegar a 2 modelos como os melhores. O primeiro possui overfitting elevado mais apresenta maior precisão, gerado pelo algoritmo do Random Forest e analisado anteriormente. Posteriormente foi utilizado a Regressão Logistica por suas caracteristicas em maior generalização e trabalho com features probabilisticas, onde encontramos o menor Log Loss, porém a precisão não se satisfez. 
# 
# Assim, não conseguimos concluir ambos os objetivos estabelecidos no mesmo modelo. Somente em modelos separados, ainda foi visto que seria possível continuar esse trabalho com novos parametros e uma maior intensidade no pre-processamento dos dados. Infelizmente devido aos limites computacionais da maquina utilizada para o projeto, não foi possivel realizar o mesmo.

# In[ ]:





# In[ ]:





# In[ ]:




