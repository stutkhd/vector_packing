import streamlit as st
import numpy as np

"""
w: アイテムの集合
B: ビンの集合
"""
def BDF(w, B, alpha=1.):
    """
    Item Based Best Fit Decreasing (BFD) heuristcs for variable size vector packing problem
    """
    # n, d = w.shape # n: 個数, d: 次元
    U = len(B) # 瓶の個数
    unassigned = []
    bins = [[] for j in range(U)]# ビンの容量
    while len(w)>0:
        W = w.sum(axis=0) #アイテムに対する残りサイズの和（次元ごと）
        R = B.sum(axis=0) #ビンに対する残り容量の和（次元ごと）

        s = (W**alpha)/(R+0.000001)**(2.-alpha) #次元の重要度

        BS = B@s #ビンのサイズ（次元統合後）
        WS = w@s #アイテムのサイズ（次元統合後）

        max_item_idx = np.argmax(WS)
        max_item = w[max_item_idx]

        for j in np.argsort(BS):
            remain = B[j]- max_item
            if np.all(remain>=0): #詰め込み可能
                B[j]= remain
                bins[j].append(max_item)
                break
        else:
            unassigned.append(max_item)
        w = np.delete(w, max_item_idx, axis=0) # 削除
    return bins, unassigned 

st.sidebar.selectbox('アルゴリズム', ['BDF'])

np.random.seed(123)
n = st.sidebar.number_input(label='アイテム個数',min_value=1, value=5)
U = st.sidebar.number_input(label='ビン数', min_value=1, value=2)
d = st.sidebar.number_input(label='考慮する要素数(次元)', min_value=1, value=2)

with st.expander("アイテムの容量制限"):
  lb = st.number_input(label='容量最小値',min_value=1, value=3)
  ub = st.number_input(label='容量最大値',min_value=1, value=10)

with st.expander("ビンの容量制限"):
  blb = st.number_input(label='容量最小値',min_value=1, value=15)
  bub = st.number_input(label='容量最大値',min_value=1, value=20)

w = np.random.randint(lb,ub,(n,d)) # アイテム配列
B = np.random.randint(blb,bub, (U,d)) # ビン配列
bins, unassigned = BDF(w, B, alpha=1.)
st.write('アイテム集合')
st.text(w)
st.write('ビン')
st.text(bins)
st.write('残りアイテム')
st.text(unassigned)