# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:13:13 2020

@author: wende
"""

# Declaração de Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from func import salvar_grafico, gerar_csv
from math import sqrt
from statistics import mean, stdev
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier

# Importação do banco de dados
dados = pd.read_csv(r"C:\Users\wende\Downloads\dadosBD2_DRONE.csv")

# Separação de atributos e labels
previsores = dados.iloc[:, 0:12].values
# classes = dados.iloc[:, 12].values

# Padronização
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Clusterização
cluster = KMeans(n_clusters=2, random_state=0, n_jobs=-1)
model = cluster.fit(previsores)
classes = model.labels_

# Criação de variáveis para gerar arquivo csv para exportação de dados
Acu_csv = pd.DataFrame()
F1_csv = pd.DataFrame()
Prec_csv = pd.DataFrame()
FPR_csv = pd.DataFrame()
FNR_csv = pd.DataFrame()
Sens_csv = pd.DataFrame()
Espec_csv = pd.DataFrame()

#Instanciamento dos classificadores
Dummy = DummyClassifier()
NaiveBayes = GaussianNB()
RegressaoLog = LogisticRegression()
SVM = SVC(kernel = 'rbf', random_state = 1, C = 2.0, probability=True)
MLP = MLPClassifier(max_iter = 500, solver='lbfgs', hidden_layer_sizes=(100), 
                     activation = 'identity', batch_size=100, learning_rate_init=0.1)
RandomForest = RandomForestClassifier(n_estimators=100, criterion='gini')


metodos = (RegressaoLog, RandomForest, MLP, SVM, NaiveBayes, Dummy)
metodos2 = ['L.Regression', 'R.Forest', 'MLP', 'SVM', 'NaiveBayes', 'Dummy']

cont = 0

int_conf = pd.DataFrame()

metricas = pd.DataFrame()

# colors = ['black', 'orange', 'blue', 'green', 'purple', 'red']
# linestyles = [':', '--', '-.', '-', ':', '--']
count = 0

for m in metodos:
    
    # Variáveis auxiliares
    acuracia_total = []
    f1_total = []
    precisao_total = []
    fpr_total = []
    fnr_total = []
    sens_total = []
    espec_total = []
    gmean_total = []
    matr1 = []
    matr2 = []
    matr3 = []
    matr4 = []
    contador = 0
    
    for c in range(30):
        # Divisão em treinamento e teste
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=c)
        
        # Variáveis auxiliares
        iteracao2 = []
        f1 = []
        precisao = []
        fnr_parc = []
        fpr_parc = []
        sens_parc = []
        espec_parc = []
        gmean_parc = []

        # Instanciamento dos classificadores
        for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(classes.shape[0], 1))):
            
            # Treinamento dos classificadores
            m.fit(previsores[indice_treinamento], classes[indice_treinamento])
            xyz = classes[indice_teste]
            previsoes = m.predict(previsores[indice_teste])
            # y_pred = m.predict_proba(previsores[indice_teste])[:, 1]
            # apr, tpr, thresholds = roc_curve(y_true=classes[indice_teste], y_score=y_pred)
        
            # Análise dos resultados
            acuracia = accuracy_score(classes[indice_teste], previsoes)*100
            matriz = confusion_matrix(classes[indice_teste], previsoes)
            f1_parcial = f1_score(classes[indice_teste], previsoes)*100
            precisao_parcial = precision_score(classes[indice_teste], previsoes)*100
            mat1 = matriz[0,0]
            mat2 = matriz[0,1]
            mat3 = matriz[1,0]
            mat4 = matriz[1,1]
            fpr = matriz[1,0]/(matriz[1,0]+matriz[1,1])*100
            fnr = matriz[0,1]/(matriz[0,1]+matriz[0,0])*100
            sens = matriz[0,0]/(matriz[0,0]+matriz[0,1])*100
            espec = matriz[1,1]/(matriz[1,1]+matriz[1,0])*100
            gmean = sqrt(sens*espec)
            
            # Preenchimento dos vetores de dados
            iteracao2.append(acuracia)
            matr1.append(mat1)
            matr2.append(mat2)
            matr3.append(mat3)
            matr4.append(mat4)
            f1.append(f1_parcial)
            precisao.append(precisao_parcial)
            fnr_parc.append(fnr)
            fpr_parc.append(fpr)
            sens_parc.append(sens)
            espec_parc.append(espec)
            gmean_parc.append(gmean)

        iteracao2 = np.asarray(iteracao2)
        media = iteracao2.mean()
        acuracia_total.append(media)
        
        f1 = np.asarray(f1)
        mediaf1 = f1.mean()
        f1_total.append(mediaf1)
        
        precisao = np.asarray(precisao)
        mediaprec = precisao.mean()
        precisao_total.append(mediaprec)
        
        fnr_parc = np.asarray(fnr_parc)
        media_fnr = fnr_parc.mean()
        fnr_total.append(media_fnr)
        
        fpr_parc = np.asarray(fpr_parc)
        media_fpr = fpr_parc.mean()
        fpr_total.append(media_fpr)
        
        sens_parc = np.asarray(sens_parc)
        media_sens = sens_parc.mean()
        sens_total.append(media_sens)
        
        espec_parc = np.asarray(espec_parc)
        media_espec = espec_parc.mean()
        espec_total.append(media_espec)
        
        gmean_total.append(mean(gmean_parc))
        
    
    # roc_auc = auc(apr, tpr)
    # plt.plot(apr, tpr, color=colors[count], linestyle=linestyles[count], label='%s (auc = %0.2f)' % (metodos2[count], roc_auc))
    # plt.legend(loc='lower right')

    Acu_csv[m] = acuracia_total
    F1_csv[m] = f1_total
    Prec_csv[m] = precisao_total
    FPR_csv[m] = fpr_total
    FNR_csv[m] = fnr_total
    Sens_csv[m] = sens_total
    Espec_csv[m] = espec_total
    matriz_final = np.array([[mean(matr1), mean(matr2)],[mean(matr3), mean(matr4)]])
    
    # Plotagem de matriz de confusão
    x = ['VM01', 'VM02']
    y = ['VM01', 'VM02']
    cm = pd.DataFrame(matriz_final, x, y)
    fig = plt.figure(figsize=(7,6))
    sns.set(font_scale=1.25)
    mapa = sns.heatmap(cm, annot=True, annot_kws={"size": 18}, cmap='Blues', 
                        linewidths=1, linecolor="black")
    mapa.set_yticklabels(mapa.get_yticklabels(), rotation=0)
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(r'C:\Users\wende\OneDrive\Imagens\Imagens Warley\matriz de confusão\{}.eps'.format(metodos2[cont]), bbox_inches="tight", dpi=100)
    cont+=1

    #Valores das métricas com intervalo de confiança
    teste = (acuracia_total, f1_total, gmean_total, precisao_total, fpr_total, fnr_total, sens_total, espec_total)
    for t in teste:
        int_conf.loc[metodos2[count], contador] = stdev(t)
        metricas.loc[metodos2[count], contador] = round(mean(t), 2)
        contador+=1
    
    count+=1
    
int_conf.columns = ['Acuracia', 'F1', 'G-mean', 'Precisao', 'FPR', 'FNR', 'Sensibilidade', 'Especificidade']
metricas.columns = ['Acuracia', 'F1', 'G-mean', 'Precisao', 'FPR', 'FNR', 'Sensibilidade', 'Especificidade']

clf1 = metricas.loc['L.Regression', 'Acuracia':'G-mean']
clf1_std = int_conf.loc['L.Regression', 'Acuracia':'G-mean']
clf2 = metricas.loc['R.Forest', 'Acuracia':'G-mean']
clf2_std = int_conf.loc['R.Forest', 'Acuracia':'G-mean']
clf3 = metricas.loc['MLP', 'Acuracia':'G-mean']
clf3_std = int_conf.loc['MLP', 'Acuracia':'G-mean']
clf4 = metricas.loc['SVM', 'Acuracia':'G-mean']
clf4_std = int_conf.loc['SVM', 'Acuracia':'G-mean']
clf5 = metricas.loc['NaiveBayes', 'Acuracia':'G-mean']
clf5_std = int_conf.loc['NaiveBayes', 'Acuracia':'G-mean']
clf6 = metricas.loc['Dummy', 'Acuracia':'G-mean']
clf6_std = int_conf.loc['Dummy', 'Acuracia':'G-mean']

x = np.arange(3)
width = 0.15

fig, ax = plt.subplots()
rects1 = ax.bar(x + 0.0, clf1, width, label='LR', yerr=clf1_std, color='Silver')
rects2 = ax.bar(x + 0.15, clf2, width, label='RF', yerr=clf2_std, color='SkyBlue')
rects3 = ax.bar(x + 0.30, clf3, width, label='MLP', yerr=clf3_std, color='MediumAquamarine')
rects4 = ax.bar(x + 0.45, clf4, width, label='SVM', yerr=clf4_std, color='DarkKhaki')
rects5 = ax.bar(x + 0.60, clf5, width, label='NB', yerr=clf5_std, color='LightSalmon')
rects6 = ax.bar(x + 0.75, clf6, width, label='DM', yerr=clf6_std, color='Tan')

ax.set_xticks(np.arange(3)+width*2.5)
ax.set_xticklabels(['Accuracy', 'F1', 'G-mean'])
ax.legend()
fig.tight_layout()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize='x-small')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

plt.savefig(r'C:\Users\wende\OneDrive\Imagens\Imagens Warley\metricas.svg', bbox_inches="tight", dpi=100)
plt.show()

# plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.grid()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.savefig(r'C:\Users\wende\OneDrive\Imagens\Imagens Warley\auc.svg', bbox_inches="tight", dpi=100)

# salvar_grafico(metricas, int_conf)
# csvs = (Acu_csv, F1_csv, Prec_csv, FPR_csv, FNR_csv, Sens_csv, Espec_csv)
# for doc in csvs:
#     gerar_csv(doc)


