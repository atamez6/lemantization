import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spacy
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

BASE = Path(__file__).parent
nlp = spacy.load("es_core_news_sm")


def lemas(tokens):
    return [t.lemma_.lower() for t in tokens if not (t.is_stop or t.is_punct or t.is_space)]


def guardar_csv(nombre, matriz, vocab):
    with open(BASE / nombre, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["documento", *vocab])
        escritor.writerows([i, *fila] for i, fila in enumerate(matriz, 1))


def limpieza(doc):
    limpio = lemas(doc)
    (BASE / "resultado_limpio.txt").write_text(" ".join(limpio), encoding="utf-8")

    print("Primeros 20 tokens originales:", [t.text for t in doc[:20]])
    print("Primeros 20 tokens limpios:", limpio[:20])
    print(f"Tokens originales: {len(doc)} | despues de la limpieza: {len(limpio)}")


def vectorizacion(texto):
    parrafos = [p.strip() for p in texto.splitlines() if p.strip()]
    docs = [lemas(d) for d in nlp.pipe(parrafos)]

    bow = CountVectorizer(analyzer=lambda x: x)
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    m_bow = bow.fit_transform(docs).toarray()
    m_tfidf = tfidf.fit_transform(docs).toarray().round(3)
    vocab = bow.get_feature_names_out()

    guardar_csv("resultado_bow.csv", m_bow, vocab)
    guardar_csv("resultado_tfidf.csv", m_tfidf, tfidf.get_feature_names_out())

    print(f"\nDocumentos: {len(docs)} | palabras en vocabulario: {len(vocab)}")
    print("Primeras palabras:", vocab[:20])
    print("BoW doc 1:", m_bow[0])
    print("TF-IDF doc 1:", m_tfidf[0])


def semantica(doc):
    oraciones = [o for o in (lemas(s) for s in doc.sents) if len(o) > 1]
    modelo = Word2Vec(oraciones, vector_size=10, window=3, min_count=1, workers=1, seed=40, epochs=200)
    modelo.save(str(BASE / "modelo_word2vec.model"))

    vocab = modelo.wv.index_to_key
    with open(BASE / "vectores_semanticos.txt", "w", encoding="utf-8") as f:
        for p in vocab:
            f.write(f"{p}: {' '.join(str(round(v, 4)) for v in modelo.wv[p])}\n")

    puntos = PCA(n_components=3).fit_transform(modelo.wv[vocab])
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*puntos.T, color="crimson")
    for palabra, (x, y, z) in zip(vocab, puntos):
        ax.text(x, y, z, palabra)
    ax.set(title="Espacio semantico del cuento", xlabel="Dimension 1", ylabel="Dimension 2", zlabel="Dimension 3")
    fig.tight_layout()
    fig.savefig(BASE / "grafica_semantica.png", dpi=150)

    print(f"\nOraciones: {len(oraciones)} | vocabulario Word2Vec: {len(vocab)}")
    for palabra in ("hormiga", "luciérnaga"):
        if palabra in modelo.wv:
            similares = [(s, round(v, 4)) for s, v in modelo.wv.most_similar(palabra, topn=3)]
            print(f"Similares a {palabra}: {similares}")


def main():
    texto = (BASE / "cuento_principito.txt").read_text(encoding="utf-8")
    doc = nlp(texto)
    limpieza(doc)
    vectorizacion(texto)
    semantica(doc)


if __name__ == "__main__":
    main()
