`numpy`・`scikit-learn`・`matplotlib` を使って、ラベルなしデータに対するクラスタリングを視覚的に理解するためのチュートリアルです。

このデモでは `make_blobs` で複数の塊を持つ 2 次元データを作り、`KMeans` で自動的にグループ分けします。正解ラベルを使わずに、データのまとまりと中心をどう推定するかを 1 枚の図で確認できます。

## What this tutorial does

- `numpy`
  - データの平均、標準偏差、各クラスタの点数を計算して図の下に表示します。
- `scikit-learn`
  - `make_blobs` で人工データを作ります。
  - `StandardScaler` でスケーリングします。
  - `KMeans` で 4 クラスタに分けます。
  - `silhouette_score` でクラスタ分離の良さを数値化します。
- `matplotlib`
  - 生データ、推定クラスタ、クラスタ領域と中心を横並びで表示します。

## How to read the output

- 左: まだラベルを付けていない元データ
- 中央: `KMeans` が割り当てたクラスタ
- 右: 各点がどの中心に最も近いかを示す領域

右端の図では、背景色が「この座標に新しい点が来たらどのクラスタに入るか」を表しています。黒い `X` はクラスタ中心で、`KMeans` が「各点を最も近い中心へ割り当てる」ことを直感的に見られます。

## Run

```bash
uv run python main.py
```

実行すると `kmeans_tutorial.png` が生成されます。

画面にも表示したい場合:

```bash
uv run python main.py --show
```
