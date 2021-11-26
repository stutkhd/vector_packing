import streamlit as st
import numpy as np
from mypulp import Model, quicksum, GRB
import pandas as pd

st.title('d次元ベクトルパッキング')

def vbpp(s,B,c,penalty=1000.):
    n = len(s)
    U = len(c)
    model = Model("bpp")
    # setParam("MIPFocus",1)
    x, y, z = {},{},{}
    for i in range(n):
        z[i] = model.addVar(vtype="B", name= f"z({i})")
        for j in range(U):
            x[i,j] = model.addVar(vtype="B", name= f"x({i},{j})")
    for j in range(U):
        y[j] = model.addVar(vtype="B", name= f"y({j})")    
    model.update()
    # assignment constraints
    for i in range(n):
        model.addConstr(quicksum(x[i,j] for j in range(U)) + z[i] == 1, f"Assign({i})")
    # tighten assignment constraints
    for j in range(U):
        for i in range(n):
            model.addConstr(x[i,j] <= y[j], f"Strong({i},{j})")
    # bin capacity constraints
    for j in range(U):
        for k in range(d):
            model.addConstr(quicksum(s[i,k]*x[i,j] for i in range(n)) <= B[j,k]*y[j], f"Capac({j},{k})")
    model.setObjective(quicksum(penalty*s[i,k]*z[i] for i in range(n) for k in range(d)) + quicksum(c[j]*y[j] for j in range(U)), GRB.MINIMIZE)

    model.update()
    model.__data = x,y,z
    return model

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

select_method = st.sidebar.selectbox('アルゴリズム', ['BDF','Normal'])

np.random.seed(123)
n = st.sidebar.number_input(label='アイテム個数',min_value=1, value=5)
U = st.sidebar.number_input(label='ビン数', min_value=1, value=3)
d = st.sidebar.number_input(label='考慮する要素数(次元)', min_value=1, value=1)

with st.expander("アイテムの容量制限"):
  lb = st.number_input(label='容量最小値',min_value=1, value=3)
  ub = st.number_input(label='容量最大値',min_value=int(lb)+1, value=10)

with st.expander("ビンの容量制限"):
  blb = st.number_input(label='容量最小値',min_value=1, value=15)
  bub = st.number_input(label='容量最大値',min_value=int(blb)+1, value=20)

if (select_method == 'Normal'):
  penalty = st.sidebar.number_input(label='ペナルティ', value=1000)

w = np.random.randint(lb,ub,(n,d)) # アイテム配列
B = np.random.randint(blb,bub, (U,d)) # ビン配列

st.write('アイテム集合')
st.text(w)
st.write('ビン集合')
st.text(B)

if select_method == 'Normal':
  c = np.random.randint(500, 1000, U)
  model = vbpp(w, B, c, penalty)
  model.optimize()
  x,y,z = model.__data
  bins = [[] for j in range(U)]
  unassigned =[]
  for i in z:
      if z[i].X > .5:
          unassigned.append(w[i])
  # bins, unassigned = 
  pass
elif select_method == 'BDF':
  bins, unassigned = BDF(w, B, alpha=1.)
st.write('ビン')
st.text(bins)
st.write('残りアイテム')
st.text(unassigned)
st.write('残ビン容量')
st.text(B)

if select_method == 'BDF' and d == 1:
  create_1D_graph(bins)