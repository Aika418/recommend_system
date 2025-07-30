import streamlit as st
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ── 読み込み ───────────────────────────
vectors = np.load("data/my_vectors.npy")                    # shape = (V, D)
vocab   = np.load("data/my_vocab.npy", allow_pickle=True)   # shape = (V,)

# ── KeyedVectors を組み立てる ─────────
kv = KeyedVectors(vector_size=vectors.shape[1])
kv.add_vectors(vocab.tolist(), vectors)      # gensim 4.x 以降の公式 API :contentReference[oaicite:0]{index=0}
kv.fill_norms()                              # 類似度計算を高速化（省略可）


st.write("マンガレコメンドアプリ")
# ── ジャンル辞書（自由に拡張してください）──────────────
genre_dict = {
    '名探偵コナン': ['推理', 'ミステリー', '学園', '日常'],
    '鬼滅の刃': ['アクション', '兄妹愛', '和風', 'バトル'],
    '呪術廻戦': ['呪術', 'バトル', 'ダークファンタジー', '学園'],
    'ワンピース': ['冒険', '海賊', '仲間', 'バトル'],
    '進撃の巨人': ['戦争', '巨人', 'ミステリー', 'ダーク'],
    '僕のヒーローアカデミア': ['ヒーロー', '学園', '能力バトル', '成長'],
    '暗殺教室': ['学園', 'コメディ', '暗殺', '感動'],
    'SPY×FAMILY': ['スパイ', '家族', 'コメディ', 'アクション'],
    'スラムダンク': ['スポーツ', 'バスケ', '青春', '成長'],
    'ブルーロック': ['サッカー', 'デスゲーム', '才能', '心理戦'],
    'ダンダダン': ['オカルト', 'SF', 'バトル', '学園'],
    'カグラバチ': ['剣術', 'バトル', '復讐', '和風ファンタジー'],
    'ゼロの日常': ['日常', '警察', 'スピンオフ', '推理'],
    '恋せよまやかし天使ども': ['恋愛', '学園', '青春', 'コメディ'],
    '薬屋のひとりごと': ['中華風', '宮廷', '推理', '医術'],
    '約束のネバーランド': ['脱出', 'サスペンス', '孤児院', 'ホラー'],
    'はたらく細胞': ['擬人化', '医療', '学習', '日常'],
    'SPY × FAMILY': ['スパイ', '家族', 'コメディ', 'アクション'],
    '犯人の犯沢さん': ['ギャグ', 'スピンオフ', 'コナン', 'ブラックコメディ']
}

# ── 新機能：類似作品 + キーワード傾向 ───────────────
def recommend_with_keywords(title, vectors, vocab, genre_dict, top_k=5):
    if title not in vocab:
        return [], []

    idx = vocab.index(title)
    similarities = cosine_similarity([vectors[idx]], vectors)[0]
    top_indices = similarities.argsort()[::-1][1:top_k+1]

    recommended_titles = [vocab[i] for i in top_indices]
    all_keywords = []
    for t in recommended_titles:
        all_keywords += genre_dict.get(t, [])
    keyword_counts = Counter(all_keywords).most_common(3)

    return recommended_titles, keyword_counts

manga_titles = vocab.tolist()

st.markdown("## 1冊のマンガに対して似ているマンガを表示する")
selected_manga = st.selectbox("マンガを選んでください", manga_titles)

if selected_manga:
    st.markdown(f"### {selected_manga} に似ているマンガ")
    results = []
    for recommend_manga, score in kv.most_similar(selected_manga, topn=30):
        results.append({"title": recommend_manga, "score": score})
    st.dataframe(pd.DataFrame(results))

    st.markdown(f"### {selected_manga} に似た作品のキーワード傾向")
    recommendations, keywords = recommend_with_keywords(selected_manga, vectors, manga_titles, genre_dict)
    if keywords:
        for kw, cnt in keywords:
            st.write(f"・{kw}（{cnt}件）")
    else:
        st.write("ジャンル情報が不足しているため傾向を表示できません。")



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


