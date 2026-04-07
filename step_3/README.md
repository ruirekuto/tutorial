`numpy`・`scikit-learn`・`matplotlib` を使って、高次元データを低次元へ圧縮する PCA を視覚的に理解するためのチュートリアルです。

このデモでは `sklearn.datasets.load_digits` の手書き数字データを使います。各画像は 8x8 ピクセルなので、1 サンプルあたり 64 次元のベクトルとして表されます。そこに `PCA` を適用して 2 次元へ圧縮し、数字ごとの分布を散布図で観察します。

## What this tutorial does

- `numpy`
  - 埋め込み後の平均座標や、主成分で説明できる分散割合を計算して図の下に表示します。
- `scikit-learn`
  - `load_digits` で手書き数字データを読み込みます。
  - `StandardScaler` で各特徴量を標準化します。
  - `PCA(n_components=2)` で 64 次元を 2 次元へ圧縮します。
- `matplotlib`
  - 元の数字画像サンプルと、2 次元に写した散布図を同じ図にまとめます。

## How to read the output

- 左側: 元の 8x8 手書き数字画像の例
- 右側: PCA によって 2 次元へ写像された全サンプル

散布図で近い位置にある点は、元の 64 次元空間でも似た特徴を持つ傾向があります。数字ごとにある程度まとまりができる一方、形が似た数字同士は重なりやすく、2 次元だけでは完全には分離できないことも見て取れます。

## Run

```bash
uv run python main.py
```

実行すると `pca_tutorial.png` が生成されます。

画面にも表示したい場合:

```bash
uv run python main.py --show
```
