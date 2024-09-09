# Análise de Dados com Pandas - Projeto Iris

- **Introdução:** Nesta análise, exploramos o dataset Iris para entender suas principais características e identificar possíveis outliers. Focamos em medidas de tendência central e visualização dos dados.
## Objetivo
Este projeto tem como objetivo realizar uma **análise exploratória de dados** utilizando o famoso dataset **Iris**. A análise envolve explorar as características morfológicas de três espécies de flores — *Setosa*, *Versicolor* e *Virginica* — com base no comprimento e na largura das sépalas e pétalas. Também iremos **remover outliers** que possam distorcer os resultados, além de visualizar os dados para identificar padrões e tendências.

## Contextualização
O dataset **Iris** é amplamente utilizado em projetos de aprendizado de máquina e análises estatísticas. Ele contém **150 registros** com quatro variáveis principais:

- **SepalLengthCm** (Comprimento da Sépala)
- **SepalWidthCm** (Largura da Sépala)
- **PetalLengthCm** (Comprimento da Pétala)
- **PetalWidthCm** (Largura da Pétala)
- **Species** (Espécie da flor)

Este projeto oferece uma excelente oportunidade para praticar a manipulação de dados com **Pandas**, a limpeza de dados e a visualização com **Matplotlib**.

## 1. Carregamento e Limpeza de Dados
Carregamos os dados diretamente da **UCI Machine Learning Repository**. Utilizamos a biblioteca **Pandas** para manipulação e a **Scipy** para detectar e remover outliers.

### Código:
```python
import pandas as pd
from scipy import stats

# Carregar o dataset Iris
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
df = pd.read_csv(url, header=None, names=column_names)

# Remover outliers utilizando o z-score
z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
df_clean = df[(abs(z_scores) < 3).all(axis=1)]

print(f"Dados antes da limpeza: {df.shape[0]} linhas")
print(f"Dados após a limpeza de outliers: {df_clean.shape[0]} linhas")
```

**Resultado**:  
Antes da limpeza: 150 linhas  
Após a remoção de outliers: 144 linhas

## 2. Análise Descritiva dos Dados
Após a limpeza, calculei as medidas de tendência central — **média**, **mediana** e **moda** — para entender melhor as características principais.

### Código:
```python
# Cálculo das medidas de tendência central
mean_values = df_clean.mean()
median_values = df_clean.median()
mode_values = df_clean.mode().iloc[0]

print("Médias:", mean_values)
print("\nMedianas:", median_values)
print("\nModos:", mode_values)
```

**Médias**:
- Comprimento da Sépala: 5.85 cm
- Largura da Sépala: 3.04 cm
- Comprimento da Pétala: 3.79 cm
- Largura da Pétala: 1.22 cm

## 3. Visualização dos Dados
Utilizei gráficos de histogramas para visualizar a distribuição das características das flores.

### Código:
```python
import matplotlib.pyplot as plt

# Gráfico de barras da distribuição das espécies
species_counts = df_clean['Species'].value_counts()
plt.figure(figsize=(10, 5))
species_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribuição das Espécies')
plt.xlabel('Espécie')
plt.ylabel('Número de Amostras')
plt.xticks(rotation=45)
plt.show()

# Histogramas das colunas numéricas
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.hist(df_clean['SepalLengthCm'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribuição do Comprimento da Sépala')
plt.xlabel('Comprimento da Sépala (cm)')
plt.ylabel('Frequência')

plt.subplot(2, 2, 2)
plt.hist(df_clean['SepalWidthCm'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribuição da Largura da Sépala')
plt.xlabel('Largura da Sépala (cm)')
plt.ylabel('Frequência')

plt.subplot(2, 2, 3)
plt.hist(df_clean['PetalLengthCm'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribuição do Comprimento da Pétala')
plt.xlabel('Comprimento da Pétala (cm)')
plt.ylabel('Frequência')

plt.subplot(2, 2, 4)
plt.hist(df_clean['PetalWidthCm'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribuição da Largura da Pétala')
plt.xlabel('Largura da Pétala (cm)')
plt.ylabel('Frequência')

plt.tight_layout()
plt.show()

```
![image](https://github.com/user-attachments/assets/886670b9-83fa-429f-934d-c64fd147a367)
![image](https://github.com/user-attachments/assets/a302aa41-7329-42a8-ba09-2828e6114006)

## 4. Insights Descobertos
- **Distribuições distintas entre espécies**: As características das pétalas e sépalas permitem diferenciar claramente as espécies.
- **Remoção de outliers**: A exclusão de dados extremos melhorou a precisão das análises.
- **Tendências visuais**: As medições das pétalas mostraram maior variação, o que pode ser útil para classificações futuras.
- **Análise Descritiva:** O comprimento médio da sépala é de 5.85 cm. A mediana é 5.8 cm, indicando que a maioria dos valores está próxima a esse número. Os modos indicam os valores mais frequentes
- **Visualização dos Dados:** Os gráficos mostram a distribuição das características das flores. Observa-se uma dispersão significativa nos comprimentos e larguras das pétalas e sépalas.
  
## Conclusão
A análise revelou que o comprimento médio da sépala é de 5.85 cm, com a maioria dos valores em torno de 5.8 cm. A distribuição dos dados de comprimento e largura das pétalas é mais dispersa. A remoção de outliers ajudou a obter uma visão mais clara das características principais.

Este projeto demonstrou a importância de uma análise cuidadosa dos dados, desde a **limpeza de outliers** até a **visualização de padrões**. O dataset Iris provou ser um ótimo recurso para prática de **Pandas** e **visualização de dados**.



