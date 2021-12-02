import streamlit as st
import numpy as np
from mypulp import Model, quicksum, GRB
import pandas as pd

st.title('d次元ベクトルパッキング')

st.header('ビンパッキング問題とは')
st.latex(r'''
          容量c(>0)の箱とn個の荷物N={1,…,n}が与えられている。\\
          荷物i∈Nの容量をw_i(>0)とする。\\
          全ての荷物を詰合わせるのに必要な箱の個数を最小にする詰合わせを求めよ。 \\
         ''')

def BDF(w, B, alpha=1.):
    """
    Item Based Best Fit Decreasing (BFD) heuristcs for variable size vector packing problem
    """
    # n, d = w.shape # n: 個数, d: 次元
    U = len(B) # 瓶の個数
    unassigned = []
    bins = [[] for j in range(U)] # ビンの容量
    while len(w)>0:
        W = w.sum(axis=0) #アイテムに対する残りサイズの和（次元ごと）
        R = B.sum(axis=0) #ビンに対する残り容量の和（次元ごと）

        s = (W**alpha)/(R+0.000001)**(2.-alpha) #次元の重要度
        
        print('s:', s)

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

def create_1D_graph(bins):
  max_dim = 0
  for bin in bins:
    max_dim = max(len(bin), max_dim)

  # padding配列を作成する
  assigned_item_total = len(bins)

  bins = [ list(map(lambda x: x.tolist(), bin)) for bin in bins]

  # paddig
  padding_bins = []
  for bin in bins:
    if len(bin) < max_dim:
      nan_arr = [[None] for _ in range(max_dim - len(bin))]
      bin.extend(nan_arr)
    padding_bins.append(bin)

  padding_bins_arr = np.array(padding_bins).reshape(assigned_item_total, max_dim)
  chart_data = pd.DataFrame(
    padding_bins_arr
  )
  st.bar_chart(chart_data)

np.random.seed(123)
n = st.sidebar.number_input(label='アイテム個数',min_value=1, value=10)
U = st.sidebar.number_input(label='ビン数', min_value=1, value=3)
d = st.sidebar.number_input(label='考慮する要素数(次元)', min_value=1, value=1)

with st.expander("アイテムの容量制限"):
  lb = st.number_input(label='容量最小値',min_value=1, value=3)
  ub = st.number_input(label='容量最大値',min_value=int(lb)+1, value=10)

with st.expander("ビンの容量制限"):
  blb = st.number_input(label='容量最小値',min_value=1, value=15)
  bub = st.number_input(label='容量最大値',min_value=int(blb)+1, value=20)

w = np.random.randint(lb,ub,(n,d)) # アイテム配列
B = np.random.randint(blb,bub, (U,d)) # ビン配列

# st.write('アイテム集合')
# st.text(w)
# st.write('ビン集合')
# st.text(B)

min_unassigned = [float('inf')]
min_bins = []
opt_alpha = float('inf')
for alpha_param in np.arange(0.0, 2.1, 0.1):
  w_copy = w.copy()
  B_copy = B.copy()
  bins, unassigned = BDF(w_copy, B_copy, alpha=alpha_param)

  print('alpha_param', alpha_param)
  print('残りビン', B)
  print(alpha_param, 'min:', np.sum(min_unassigned), 'now:', np.sum(unassigned))
  if np.sum(min_unassigned) > np.sum(unassigned):
    print('hello')
    min_unassigned = unassigned
    min_bins = bins
    opt_alpha = alpha_param
print('alpha_param = 0:', np.sum(BDF(w, B.copy(), alpha=0)[1]))
print('alpha_param = 1:', np.sum(BDF(w, B.copy(), alpha=1.0)[1]))
print('alpha_param = 2.0:', np.sum(BDF(w, B.copy(), alpha=2.0)[1]))
# bins, unassigned = BDF(w, B, alpha=1.)

# st.write('最小積み残しの時のalpha')
# st.text(opt_alpha)
# st.write('最小積み残しの時のビンの詰め方')
# st.text(min_bins)
# st.write('最小残りアイテム')
# st.text(min_unassigned)
# st.write('残りアイテムの合計')
# st.text(sum(min_unassigned))

# if d == 1:
#   create_1D_graph(bins)