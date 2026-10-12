import spacy
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

nlp = spacy.load("es_core_news_sm")

with open("cuento_principito.txt", "r", encoding="utf-8") as f:
    doc = nlp(f.read())

sentences = []
for sent in doc.sents:
    tokens = [
        token.lemma_.lower()
        for token in sent
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    if len(tokens) > 1:
        sentences.append(tokens)

print(f"Total de oraciones procesadas: {len(sentences)}")

model = Word2Vec(
    sentences,
    vector_size=10,
    window=5,
    min_count=1,
    workers=multiprocessing.cpu_count(),
    seed=42
)


def mostrar_similares(palabra):
    try:
        similares = model.wv.most_similar(palabra, topn=3)
        print(f"\n{palabra}:")
        for sim in similares:
            print(f"  - {sim[0]} ({sim[1]:.4f})")
    except KeyError:
        print(f"\n'{palabra}' no está en el vocabulario")


mostrar_similares("principito")
mostrar_similares("rosa")
mostrar_similares("zorro")

vocabulario = model.wv.index_to_key[:60]
vectores = model.wv[vocabulario]

pca = PCA(n_components=3)
vectores_3d = pca.fit_transform(vectores)

df_3d = pd.DataFrame(vectores_3d, columns=['x', 'y', 'z'])
df_3d['palabra'] = vocabulario

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df_3d['x'], df_3d['y'], df_3d['z'], c='crimson', s=80, edgecolors='white', alpha=0.8)

for i, row in df_3d.iterrows():
    ax.text(row['x'], row['y'], row['z'], f" {row['palabra']}", size=10)

ax.set_title('Espacio Semántico (Word Embeddings) - cuento el Principito', fontsize=14)
ax.set_xlabel('Dimensión Latente 1')
ax.set_ylabel('Dimensión Latente 2')
ax.set_zlabel('Dimensión Latente 3')

plt.tight_layout()
plt.savefig("espacio_word2vec.png", dpi=150)
plt.show()