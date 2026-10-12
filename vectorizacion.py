import spacy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA

nlp = spacy.load("es_core_news_sm")

with open("cuento_principito.txt", "r", encoding="utf-8") as f:
    doc = nlp(f.read())

corpus_lematizado = []

for oracion in doc.sents:
    lemas_oracion = [
        token.lemma_.lower()
        for token in oracion
        if not token.is_punct and not token.is_space and not token.is_stop
    ]
    if lemas_oracion:
        corpus_lematizado.append(" ".join(lemas_oracion))

print(f"Total de oraciones procesadas: {len(corpus_lematizado)}")

with open("texto_procesado.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(corpus_lematizado))


def graficar_palabras_3d(ax, matriz, vocabulario, titulo, color_puntos, n=60):
    matriz_palabras = matriz.T

    idx = np.asarray(matriz_palabras.sum(axis=1)).ravel().argsort()[::-1][:n]
    matriz_palabras = matriz_palabras[idx]
    vocabulario = vocabulario[idx]

    pca = PCA(n_components=3)
    coords = pca.fit_transform(matriz_palabras.toarray())

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    ax.scatter(x, y, z, c=color_puntos, s=80, edgecolors='k', alpha=0.8, depthshade=True)

    for i, palabra in enumerate(vocabulario):
        ax.text(x[i], y[i], z[i] + 0.1, palabra, fontsize=9)

    ax.set_title(titulo)
    ax.set_xlabel('Comp. Principal 1')
    ax.set_ylabel('Comp. Principal 2')
    ax.set_zlabel('Comp. Principal 3')

    ax.plot([0, 0], [0, 0], [z.min(), z.max()], c='grey', ls='--', lw=0.5, alpha=0.3)
    ax.plot([x.min(), x.max()], [0, 0], [0, 0], c='grey', ls='--', lw=0.5, alpha=0.3)
    ax.plot([0, 0], [y.min(), y.max()], [0, 0], c='grey', ls='--', lw=0.5, alpha=0.3)


fig = plt.figure(figsize=(18, 8))

ax1 = fig.add_subplot(121, projection='3d')

bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(corpus_lematizado)
vocab_bow = bow_vectorizer.get_feature_names_out()

graficar_palabras_3d(ax1, X_bow, vocab_bow, "Espacio BoW 3D (Conteos)", "orange")

ax2 = fig.add_subplot(122, projection='3d')

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)
vocab_tfidf = tfidf_vectorizer.get_feature_names_out()

graficar_palabras_3d(ax2, X_tfidf, vocab_tfidf, "Espacio TF-IDF 3D (Importancia)", "teal")

plt.tight_layout()
plt.savefig("espacio_bow_tfidf.png", dpi=150)
plt.show()