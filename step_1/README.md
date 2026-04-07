`numpy`・`scikit-learn`・`matplotlib` を使って、線形モデルと非線形モデルの違いを視覚的に理解するためのチュートリアルです。

このデモでは `sklearn.datasets.make_moons` で 2 クラスの曲線的なデータを作り、`LogisticRegression` と `KNeighborsClassifier` の振る舞いを比較します。線形分離が苦手なデータに対して、モデルによって決定境界がどう変わるかを 1 枚の図で確認できます。

## What this tutorial does

- `numpy`
  - 学習データの平均や広がりを計算して、図の下に要約を表示します。
- `scikit-learn`
  - `make_moons` でデータを作成します。
  - `train_test_split` で訓練用とテスト用に分けます。
  - `LogisticRegression` と `k-NN` を学習し、精度を比較します。
- `matplotlib`
  - 訓練データ、テストデータ、2 つのモデルの決定境界を 2x2 で可視化します。

## How to read the output

- 左上: 学習に使うデータ
- 右上: 評価に使うテストデータ
- 左下: ロジスティック回帰による線形の決定境界
- 右下: k-NN による非線形の決定境界

`make_moons` のような曲がった構造を持つデータでは、線形モデルは単純な直線的境界しか作れません。一方で k-NN は局所的な構造を反映しやすく、より自然な分類領域を作れることが分かります。

## Run

```bash
uv run python main.py
```

実行すると `moon_tutorial.png` が生成されます。

画面にも表示したい場合:

```bash
uv run python main.py --show
```
