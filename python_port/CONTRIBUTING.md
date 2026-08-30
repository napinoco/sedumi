# sedumi-py 開発ガイド(作業方針)

このドキュメントは、SeDuMi の MATLAB/Octave 非依存移植プロジェクトに
新しく参加する人(人間・AIエージェント問わず)が迷わず作業を続けられる
ようにするための引き継ぎ資料です。「何が終わっていて、何が残っていて、
どういう手順・方針で進めてきたか」をまとめています。

## 1. プロジェクトのゴール

SeDuMi(MATLAB/Octave 上で動く SDP/SOCP 用の内点法ソルバー、`.m` ファイル
約96個 + MEX用 `.c` ファイル約54個)を、**MATLAB にも Octave にも一切
依存しない**形に移植する。採用したアーキテクチャ(過去のセッションで
決定済み、"Option B"):

- アルゴリズム本体(内点法の反復ロジック)→ **Python (NumPy/SciPy)**
- 性能が重要な低レベルカーネル(Cholesky分解、コーン演算など)→
  既存の `.c` ファイルを MEX 依存を取り除いて**独立 C ライブラリ**
  (`libsedumi.so`)としてビルドし、**ctypes** 経由で Python から呼ぶ

最終的には `pip install` できる、MATLAB/Octave 不要の Python パッケージ
にすることが目標。

## 2. 全体のフェーズと現在の進捗

タスク管理ツール(TaskList)にフェーズごとのタスクが登録されている。
2026-08-30 時点の状況:

| フェーズ | 内容 | 状態 |
|---|---|---|
| Phase 0 | 検証基盤(Octave実機でgolden reference取得) | **完了** |
| Phase 1 | Cカーネルのmex依存除去→独立Cライブラリ化 | **完了** |
| Phase 2 | Pythonバインディング(ctypes)構築 (クラスタ1〜5) | **完了** |
| Phase 3-a | 薄いMEXラッパー`.m`をPython APIとして整備 | **未着手** |
| Phase 3-b | コーン数学ユーティリティ(eigK, psdeig, psdscale等)移植 | **完了** |
| Phase 3-c | 内点法の反復制御ロジック(sdinit〜optstep)移植 | **完了** |
| Phase 3-d | `sedumi.m`本体の移植 + golden referenceでの全体検証 | **完了(LP+SOCPスコープ)** |
| Phase 4 | 高レベルAPI・入出力互換層(.mat/SDPA)の実装 | **未着手** |
| Phase 5 | 検証・ベンチマーク | **未着手** |
| Phase 6 | パッケージング・リリース | **未着手** |

Phase 3(内点法アルゴリズム本体の移植)は **LP + SOCP(2次錐)問題に
限っては完了**しており、実際に `sedumi_port.sedumi.sedumi(A,b,c,K)` を
呼べば Octave 版の SeDuMi と完全一致する解が返ってくることを、実機
オラクル比較で確認済み(`python_port/tests/test_sedumi.py`)。

**ただし PSD 錐(`K.s`)と密列(dense columns)最適化は未対応** で、
これが現時点でのスコープ上の最大の制約。詳細は §5, §7 参照。

## 3. ディレクトリ構成

```
sedumi/                      # リポジトリルート(元のMATLAB/Octave版 .m/.c 一式はここに残っている)
  python_port/
    sedumi_port/              # 移植先の Python パッケージ本体
      _native.py               # ctypes バインディング集約(全Cカーネル呼び出しはここに集める)
      libsedumi.so              # ビルド済み共有ライブラリ(tools/build_libsedumi.sh で生成)
      cone.py                   # コーン数学ユーティリティ(eigK, psdeig, psdscale, frameit...)
      pretransfo.py / posttransfo.py   # 外部形式 <-> 内部形式の変換
      sdinit.py                 # 初期点生成
      sdfactor.py / sddir.py    # 自己双対埋め込みの分解・方向計算
      pcg.py                    # 前処理付き共役勾配法(loopPcg/wrapPcg)
      wregion.py                # 1反復分の predictor(+corrector)ステップ本体
      updtransfo.py             # スケーリング点の更新
      maxstep.py / widelen.py / stepdif.py / trydif.py  # ステップ長計算
      getada.py / getdatm.py / deninfac.py  # ADA行列の構築・分解
      symbchol.py                # ADAの記号的コレスキー(一度だけ実行)
      optstep.py                 # LP最適性の早期判定(optstep.m)
      amul.py / checkpars.py     # 補助ユーティリティ
      sedumi.py                  # トップレベルドライバ(全部をつなぐ)
    tests/
      test_*.py                  # 各モジュールの検証テスト(オラクル比較)
      fixtures/                  # Octave実機で生成した .mat オラクルデータ(コミット済み)
      golden/                    # Phase 0 の golden reference
    tools/
      generate_*_oracle.m        # 各テストのオラクルを実際のOctave/MEXビルドで生成するスクリプト
      build_libsedumi.sh         # libsedumi.so のビルドスクリプト
    README.md                    # (やや古い。フェーズ概要はこのCONTRIBUTING.mdの方が新しい)
```

## 4. 開発ワークフロー(1つの `.m` ファイルを移植する際の手順)

これまで一貫して踏襲してきた手順。新しく関数を移植するときはこれに
従うこと。

1. **対象の `.m` ファイルの実ソースを全部読む。** コメントだけでなく
   実装ロジックを1行ずつ理解する。thin MEX wrapper(`sedumi_binary_error()`
   だけを呼ぶスタブ)の場合は対応する `.c` ファイルを読む。
2. **忠実な Python 移植を書く。** 変数名や処理順序はできるだけ `.m`/`.c`
   に対応させ、後から見比べやすくする。ドキュストリングには「何を
   ポートしたか」「`.m` ファイルの何行目に対応するか」「意図的に省略
   した部分とその理由」を明記する。
3. **Octave実機でオラクルを生成する。** `python_port/tools/generate_<name>_oracle.m`
   を書き、リポジトリルートの実際の `.m`/MEX ビルド(`install_sedumi` 済み)
   を呼び出して、入力データと出力を `.mat` に保存する。
   ```
   octave-cli --no-gui --eval "cd python_port/tools; generate_<name>_oracle"
   ```
   既存フィクスチャ(例: `pretransfo` の `K2`/`prep`)を使い回せる場合は
   積極的に再利用し、ゼロから計算し直すコストを避ける。
4. **`test_<name>.py` を書いて比較する。** `.mat` を読み込み、Python版の
   出力と `np.testing.assert_allclose` で突き合わせる。
5. **不一致が出たら中間値を1つずつ突き合わせてデバッグする。** 「なんと
   なく直す」のではなく、Octave側とPython側で同じ変数を1つずつ出力して
   どこで最初にズレるかを特定する。これまで見つかった実バグ(§6参照)
   は全てこの方法で見つけている。
6. **`python_port/tests/ -q` を全部流して回帰がないか確認してからコミット。**
   コミットメッセージには「何を移植したか」「見つけたバグとその修正」
   「意図的なスコープ制限とその理由」を書く。

### オラクルスクリプトを書くときの注意(乱数ストリーム)

Octave の `rand('seed', N)` は一様乱数の状態しかリセットしない。
`randn` は別ストリームなので、`randn` を使うケースでは必ず
`randn('seed', N)` も一緒に呼ぶこと(呼び忘れると実行するたびに
別のデータが生成され、フィクスチャが再現不能になる → 実際にこの
セッションでハマった)。

また、既存のオラクル生成スクリプトの**途中**に新しいコードを挿入すると、
それより後ろの `rand()` 呼び出しが全部ズレて、無関係な既存フィクスチャ
まで書き換わってしまうことがある。新しいケースは必ずスクリプトの
**末尾に追記**し、`git diff --stat` で既存フィクスチャのファイルサイズが
変わっていないか確認すること。

## 5. スコープ上の制約(意図的に未実装にしている部分)

以下は「気づいていない抜け」ではなく、**意図的に `NotImplementedError`
で弾いている**制約。理由も含めて各モジュールのdocstringに明記済み。

- **PSD錐(`K.s`)のメインループ対応。** `sedumi.m` 本体のADA更新は
  `sum(K.s)==0` で分岐しており、この移植(`getada.py`)は `==0` の
  分岐しか実装していない。`!=0` 側は `getada1`/`getada2`/`getada3`
  (Cバインディングとしては Phase 2 クラスタ4で既にctypes化済み)の
  オーケストレーションが必要で、まだ Python 側の結線ができていない。
  → `sedumi.py` は `K.s` が非空だと `NotImplementedError` を投げる。
- **密列(dense columns)最適化。** `getdense.m`(密列検出)、
  `incorder.m`(貪欲順序付け)、`getsymbada.m`、`symbcholden.m` は
  未移植。`dense.cols`/`dense.q` は常に空として扱う。これは**性能
  最適化であって正解性には無関係**(密列を分離してもしなくても
  `A*P(d)*A'` という同じ線形方程式を解くだけ)なので、正解性は
  損なわれないが、密列を大量に含む問題では収束が遅くなったり数値的に
  不安定になったりする可能性がある。`getdatm.py`/`pcg.py`/`deninfac.py`
  が一貫して `dense.cols`/`dense.q` 非空時に `NotImplementedError`
  を投げるようにしてあるので、grep すればスコープの境界が分かる。
- **回転2次錐(`K.r`)** は `pretransfo.py` の段階で標準2次錐(`K.q`)に
  変換されるため、それより後段のコードは意識する必要はない
  (`sedumi.py` の検証テストにも回転錐のケースが1つ含まれている)。
- **コンソール出力・v-plot・`pars.stopat` のデバッグブレーク・事前の
  ランク診断・DIMACS誤差指標(`info.err`)** は移植していない。これらは
  全て「診断・表示専用」で `(x,y,info)` の値そのものには影響しない
  ため、優先度を下げている。

## 6. これまでに見つかった実バグ・注意点(教訓)

- **MATLABの値渡し意味論 vs ctypesのin-place変更。** `fwsolve`/`bwsolve`
  はCカーネル(`fwblkslv.c`/`bwblkslv.c`)がin-placeで書き換える設計を
  そのままctypesバインディングしているため、呼び出し側でバッファを
  使い回すMATLABコードをそのまま移植すると**サイレントに壊れる**。
  `pcg.py` の `sparfwslv`/`sparbwslv` で「呼び出し前に必ずコピーする」
  ように統一して解決した。同種の罠は他のin-placeネイティブ関数
  (`fwdpr1`/`bwdpr1`等)にもあり得るので、新しくバインディングを
  使うときは常に疑うこと。
- **mexFunctionの自動スライシング。** `psdframeit.c`/`psdinvjmul.c` の
  ような一部のMEXカーネルは、「PSD部分だけの短い配列」と「L+Q+PSDの
  フル長配列」の両方を受け付け、フル長の場合は自動でオフセットを
  読み飛ばす(`x += cK.lpN + 2*cK.lorN`)。Phase 2のctypesバインディング
  がこの自動スライシングを再現し忘れていたことが `psdinvjmul` で
  実際に発覚した(`wregion.py` 移植時)。新しいバインディングを
  追加する際は元の `mexFunction` のこの種の分岐を見落とさないこと。
- **qsortコンパレータの未定義動作。** `sortnnz.c`/`iswnbr.c` は
  `signed char` を返すコンパレータを `int(*)(const void*,const void*)`
  にキャストして `qsort()` に渡しており、実機で実際に非決定的な
  挙動を確認した。この種の関数は**ctypesバインディングせず**、
  コメントに書かれた本来のアルゴリズムをPythonで直接書き直す方針
  にしている(`neighborhood.py` の `iswnbr` が実例)。
- **`symbchol.m` の完全密行列分岐。** ADAが完全に密(全要素非ゼロ)の
  場合、実際の `symbchol.m` は最小次数順序付け(MMD)を省略して恒等
  順序+単一巨大supernodeを直接使う。これを再現せず常に `ordmmd` を
  呼ぶと、**収束はするが反復回数がOctave版とズレる**(小さい密な
  テスト問題で実際に確認済み)。`symbchol.py` はこの分岐を正確に
  再現している(`_native.symbolic_cholesky_dense`)。
- **MATLABの `'` は共役転置。** 実数配列に対しても、複素数が絡む式
  (`posttransfo.m` の `(x'*prep.QR)'` 等)では単純転置と共役転置の
  違いが結果に影響する。移植時は常に注意する。
- **`optstep.m` の `sum(K.s)!=0` 分岐は実質デッドコード。** `sedumi.m`
  が `optstep` を呼ぶのは `lponly = (K.l==length(c))` の場合のみで、
  これは `K.q`/`K.s` が両方空であることを強制する。つまり
  `optstep.m` 自身にPSD対応の分岐が書いてあっても、実際の呼び出し
  経路からは絶対に到達しない。こういう「到達不能性を呼び出し元の
  条件から証明できるデッドコード」は、移植せずに `NotImplementedError`
  で塞いで良い、という判断をしている。ただし本当に呼び出し元の
  条件を確認してからにすること(推測で「多分デッドコードだろう」で
  済ませない)。

## 7. 残っている作業(優先度が高いと思われる順)

1. **Phase 3-a: 薄いMEXラッパー`.m`の公開API整備。** `_native.py` に
   ctypesバインディングとしてはあるが、`sedumi_port` パッケージの
   公開APIとして整理されていない関数がないか棚卸しする。
2. **PSD錐(`K.s`)のメインループ対応。** `getada1`/`getada2`/`getada3`
   のPython結線 + `Aord`(`sortnnz`/`incorder` ベース)の構築が必要。
   `incorder.c` はqsort UBを含まない決定的アルゴリズムなので、
   `neighborhood.py` と同じ方針(Cソースを直接Pythonに書き写す)で
   移植できるはず(このセッションで既にソースは読んで解析済み、
   まだファイルには書き出していない)。
3. **密列(dense columns)最適化。** `getdense.py`/`incorder.py`/
   `getsymbada.py`/`symbcholden.py` の移植 + `adendotd`/`adenscale`/
   `dpr1fact` のオーケストレーション。正解性には影響しないため優先度
   はPSD対応より低い。
4. **Phase 4: 高レベルAPI・入出力互換層。** `.mat`/SDPA形式の読み書き、
   `sedumi()` のPython的に自然なシグネチャ・引数バリエーション対応
   (現状 `sedumi(A,b,c,K,pars=None)` の単純な形のみ)。
5. **Phase 5: 検証・ベンチマーク。** Phase 0 の golden reference
   (`tests/golden/`)に対する網羅的な回帰テスト、性能比較。
6. **Phase 6: パッケージング。** `cibuildwheel` 等でのwheel化、
   `libsedumi.so` の同梱方法の検討(現状はビルド済みバイナリを
   そのままリポジトリに置いている)。

## 8. テストの実行方法

```
cd /home/user/sedumi
.venv/bin/python -m pytest python_port/tests/ -q
```

Octaveがインストールされていない環境でもテストは通る(オラクル
データは `.mat` として事前生成・コミット済みで、テストは実行時に
Octaveを呼ばない)。オラクルを**再生成**する場合のみ Octave
(`install_sedumi` 実行済みの状態)が必要:

```
octave-cli --no-gui --eval "cd python_port/tools; generate_<name>_oracle"
```

## 9. コーディング規約・命名の慣習

- **K(コーン構造体)は Python の `dict`** で表現する。フィールド名は
  `.m` 版の `K.f`/`K.l`/`K.q`/... とできるだけ同じにする
  (`K["l"]`, `K["mainblks"]` など)。`pretransfo.py` が計算する
  内部専用フィールド(`mainblks`, `qblkstart`, `sblkstart`, `lq`,
  `N`, `rsdpN` 等)はそのまま下流の全関数に引き回す。
- **インデックスは基本 0-indexed(Python流)**。ただし `K` の中身
  (`mainblks` 等)は元の `.m` の1-indexed値をそのまま持っていることが
  多く、使う側で `int(x) - 1` のような変換をその都度行っている
  (統一的な変換層はまだ無い)。新しいコードを書くときは既存の
  近い関数のインデックス変換パターンを真似ること。
- **疎行列は `scipy.sparse.csc_matrix`** を基本形式とする。
- **全てのCカーネル呼び出しは `_native.py` に集約**し、他のモジュール
  は `_native.py` 経由でのみCコードに触れる。
- **各moduleのdocstringに「何を実装し、何を意図的に実装していないか」
  を明記する**のがこのプロジェクトの一貫した文化。読んだだけで
  スコープが分かるようにしておくこと。
