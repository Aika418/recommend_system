import streamlit as st
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
import numpy as np

# ── 読み込み ───────────────────────────
vectors = np.load("data/my_vectors.npy")                    # shape = (V, D)
vocab   = np.load("data/my_vocab.npy", allow_pickle=True)   # shape = (V,)

# ── KeyedVectors を組み立てる ─────────
kv = KeyedVectors(vector_size=vectors.shape[1])
kv.add_vectors(vocab.tolist(), vectors)      # gensim 4.x 以降の公式 API :contentReference[oaicite:0]{index=0}
kv.fill_norms()                              # 類似度計算を高速化（省略可）


st.write("マンガレコメンドアプリ")

manga_titles = vocab.tolist()

st.markdown("## 1冊のマンガに対して似ているマンガを表示する")
selected_manga = st.selectbox("マンガを選んでください", manga_titles)

# 似ている映画を表示
st.markdown(f"### {selected_manga}に似ているマンガ")
results = []
for recommend_manga, score in kv.most_similar(selected_manga,topn=30):
    results.append({"title": recommend_manga, "score": score})
results = pd.DataFrame(results)
st.write(results)


# 本来なら下記のような簡単な読み込みで対抗可能。ただし、gensimバージョンが異なるとエラー出る
# model = gensim.models.word2vec.Word2Vec.load("data/manga_item2vec.model")

st.markdown("## 複数のマンガを選んでおすすめのマンガを表示する")
selected_mangas = st.multiselect("マンガを複数選んでください", manga_titles)

if selected_mangas:
    vectors = [kv.get_vector(m) for m in selected_mangas if m in kv]
    if len(vectors) > 0:
        user_vector = np.mean(vectors, axis=0)
        st.markdown("### あなたにおすすめのマンガ")
        recommend_results = []
        for manga_title, score in kv.most_similar(user_vector, topn=30):
            if manga_title not in selected_mangas:  # 選んだものは除外
                recommend_results.append({"title": manga_title, "score": score})
        st.dataframe(pd.DataFrame(recommend_results))


