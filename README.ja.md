<h1 align="center">oumu</h1>

<p align="center">Rust と CUDA で書いた <a href="https://github.com/OpenBMB/VoxCPM">VoxCPM2</a> 音声合成エンジン</p>

<p align="center"><a href="#クイックスタート"><b>クイックスタート</b></a> · <a href="#使い方">使い方</a> · <a href="#仕組み">仕組み</a> · <a href="#検証">検証</a> · <a href="#制限">制限</a></p>

<p align="center">
  <a href="https://github.com/hayatree/oumu/actions/workflows/ci.yml"><img src="https://github.com/hayatree/oumu/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#ライセンス"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue" alt="License: MIT OR Apache-2.0"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <b>日本語</b></sub></p>

テキストと、必要なら声の見本の録音を渡すと、48 kHz の WAV を書き出します。

- **テイクをまとめて生成**：1つのテキストの複数テイク（シードごとに1本）を1バッチでデコードします。候補をいくつか作って、良いものを選べます。
- **本家に忠実**：プロンプトのトークン列は `voxcpm` と同じで、AudioVAE は PyTorch との差が 1e-5 未満です。評価したサンプルでは、音声の指標も本家と近い値でした（「[検証](#検証)」を参照）。
- **1つのバイナリ、依存は1つ**（[cudarc](https://github.com/coreylowman/cudarc)）：実行時に Python は使いません。チェックポイントの pickle は、中のコードを実行せずに読みます。

現状は 0.1 です。声を複製する前に「[責任ある利用](#責任ある利用)」を読んでください。

## クイックスタート

Rust、CUDA ツールキット、NVIDIA GPU（詳しくは「[ビルド](#ビルド)」）と、モデルのダウンロード用に [uv](https://docs.astral.sh/uv/) があれば、次の手順で動きます。

    git clone https://github.com/hayatree/oumu && cd oumu
    cargo build --release --locked

    uvx --from huggingface_hub hf download openbmb/VoxCPM2 \
        model.safetensors audiovae.pth config.json tokenizer.json tokenizer_config.json \
        --local-dir models/VoxCPM2

    ./target/release/oumu --model models/VoxCPM2 --text "こんにちは、oumu です。" --out hello.wav

録音を渡さなければ、声はモデルが決めます。声の複製は「[使い方](#使い方)」を見てください。

## ビルド

必要なもの：Linux、Ampere 世代以降の NVIDIA GPU（sm_80 以上、空きメモリ約7 GB）、cudarc 0.19 が対応する CUDA ツールキット（`nvcc` を使用し、cuBLAS は実行時に読み込み）、Rust 1.88 以上。

    cargo build --release --locked      # 実行ファイルは target/release/oumu

| 変数 | 既定値 | 内容 |
|---|---|---|
| `NVCC`, `CUDA_HOME` | `PATH` 上の `nvcc`、なければ `/usr/local/cuda` | カーネルをコンパイルする CUDA コンパイラ。cudarc は `CUDARC_CUDA_VERSION` がなければ `PATH` 上の `nvcc --version` を使います |
| `CUDA_ARCH` | `compute_80` | PTX のアーキテクチャ。実行する GPU 向けにドライバが JIT コンパイルします |
| `CUDA_SASS` | なし | 指定した GPU 向けの機械語も埋め込み、JIT を省きます（例：`86,89,121`）。コンテナで便利です |
| `CUDARC_CUDA_VERSION` | `nvcc --version` から判定 | バインディングの CUDA API の版（例：`12080`） |

NVIDIA DGX Spark（GB10、sm_121、CUDA 13.0、aarch64）で開発・動作確認しています。CI では x86-64 と CUDA 12.8 の環境（GPU なし）でビルドと CPU のテストを行っています。他の GPU での報告を歓迎します。

## 使い方

モデルのディレクトリには Hugging Face のスナップショット [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) を指定します（`model.safetensors`、`audiovae.pth`、`config.json`、`tokenizer.json`、`tokenizer_config.json` を読みます）。

    # 録音の声で1テイク
    oumu --model ~/models/VoxCPM2 --ref me.wav --prompt me.wav --prompt-text "me.wav で話している内容。" \
        --text "エンジンからこんにちは。" --out hello.wav

    # 複数のテキストを4テイクずつ：takes/cand-<id>-s<seed>.wav
    oumu --model ~/models/VoxCPM2 --ref me.wav --prompt me.wav --prompt-text "..." \
        --texts texts.json --out-dir takes --prefix cand- --seeds 7,11,23,42 --trim

| 声のオプション | 効果 |
|---|---|
| なし | 声はモデルが決めます |
| `--ref R.wav` | R の声を複製します |
| `--prompt P.wav --prompt-text T` | 録音 P の続きとして話します（T は P の正確な書き起こし）。話し方が最も近くなります |
| 3つすべて（同じファイルを2回） | 最も似せる方法です（VoxCPM2 の作者の推奨） |

録音：WAV（16/24/32ビット PCM または32ビット float）、チャンネル数は問いません（混ぜてモノラルに）、1〜768 kHz。テキストの先頭に `(少し速く、明るく)` のような指示を書くと、`voxcpm` と同じように効きます。

| オプション | 内容 |
|---|---|
| `--text TEXT --out FILE.wav` | テキスト1つ。シードが複数なら `FILE-s<seed>.wav` |
| `--texts FILE.json --out-dir DIR [--prefix STR]` | `{"id": "テキスト", ...}` または `["テキスト", ...]` → `DIR/<prefix><id>[-s<seed>].wav` |
| `--seeds LIST` | カンマ区切りのシード。1つにつき1テイク（既定 `1`、最大256） |
| `--cfg X` | 分類器なしガイダンスの強さ（既定 2.0） |
| `--timesteps N` | パッチごとの拡散ステップ数、2〜100（既定 10） |
| `--min-len N` | 1テイクは最低 N＋2 パッチ（1パッチ0.16秒、既定 2。`voxcpm` と同じ） |
| `--max-len N` | 1テイクの最大パッチ数（既定 4096）。`voxcpm` と同じくテキストのトークン数×6＋10 も上限で、KV キャッシュの空きも超えません |
| `--trim` | 前後の無音（ピークの2%未満）を50ミリ秒残して切ります |
| `--float` | 16ビット PCM の代わりに32ビット float の WAV |
| `--kv N` | テイクごとの KV キャッシュ長＝プロンプト＋生成パッチ（既定 4096。64の倍数に切り上げ） |
| `--max-prompt N` | プロンプトの最大長（既定 1024。13秒の録音を2回使い1文足すと約330） |
| `--quiet` | 標準エラーに進み具合や時間を出しません |

## 仕組み

| 段階 | ファイル | 内容 |
|---|---|---|
| テキスト | `tok.rs`、`prompt.rs` | `transformers` が読み込むのと同じトークナイザ、VoxCPM の漢字分割、`_generate_with_prompt_cache` と同じプロンプト組み立て |
| 声 | `audio.rs`、`vae.rs` | WAV → 16 kHz → AudioVAE エンコーダ（fp32。cuBLAS の行列積と、depthwise 畳み込み＋Snake をまとめたカーネル） |
| デコード | `model.rs`、`kernels/voxcpm.cu` | プレフィルは1回だけ行い、その状態を各テイクのスロットへコピー。以降は1ステップ1つの CUDA グラフ。bf16 で保持し fp32 で計算、PyTorch と同じ箇所で丸め |
| 波形 | `vae.rs`、`kernels/vae.cu` | AudioVAE デコーダ（fp32）。本家と同じく、プロンプトの最後のパッチを前に付けてデコードし、切り落とします |
| 読み込み | `st.rs`、`pth.rs` | safetensors をメモリマップして GPU へ転送。`audiovae.pth` はテンソルと辞書しか扱えない pickle インタプリタで読みます |

`tts.rs` がこれらをまとめ（`Tts::load`、`voice`、`speak`）、`main.rs` がコマンドライン、`dev.rs` が下の照合を担当します。

デコードの1ステップは、読み込む重みの量に比べて計算が少ないため、読むメモリとカーネルの起動回数を減らすように作っています。

- 1ステップ全体（ガイダンス付きの拡散パス、ローカルエンコーダ、停止判定、2つの言語モデル）を一度 CUDA グラフに記録し、毎ステップ1回の起動で再生します。
- 注意計算は KV キャッシュの埋まっている部分だけを読みます。
- 小さな行列積では、各重みをテイクごとではなく、バッチ内の全テイク分まとめて1回だけ読みます。

## 検証

各段階を Python 実装（`voxcpm`、PyTorch 2.10）と比べました。13秒の録音1本を reference と prompt の両方に使い、日本語のナレーションを生成しています。

| 段階 | 比較相手 | 結果 |
|---|---|---|
| トークナイザ | `transformers`、56文（漢字・かな、韓国語、アクセント、絵文字、数字、空白、特殊トークン） | 55文が一致。残る1文は `<\|im_end\|>` という文字列をそのまま含み、`transformers` が自身の設定と食い違う ID に変換するもの |
| プロンプト | `voxcpm` | トークン列、マスク、生成の上限が一致 |
| 16 kHz へのリサンプリング | librosa（soxr HQ） | 相対 L2 誤差 9e-5 |
| AudioVAE エンコーダ／デコーダ | 厳密な fp32 の PyTorch | 相対 L2 誤差 3.5e-6／1.3e-6（PyTorch 自身も既定の TF32 では 4e-3／7e-4 ずれる） |
| デコード（本家と同じ乱数を再生） | PyTorch の bf16 の途中経過 | プレフィル後と最初の拡散出力で相対 L2 誤差 1e-2（bf16 の丸め誤差の大きさ）。止まるステップも同じ |

その後のパッチでは bf16 の丸め誤差が積み重なり、ずれが大きくなることがあります。bf16 の実装どうしでは起こりうることなので、出力の品質で比べました。両方の実装で32テイク（8テキスト×4シード）を作り、同じ方法で評価した結果です。

| | `voxcpm` | oumu |
|---|---|---|
| 文字誤り率（Whisper large-v3 で書き起こし） | 0.084 | 0.086 |
| 見本の声との高さの差 | 0.9半音 | 1.3半音 |
| 話す速さ | 毎秒7.34文字 | 毎秒7.42文字 |
| スペクトル重心 | 911 Hz | 911 Hz |

入力、オプション、GPU、ビルドが同じなら、同じシードの組で何度実行してもビット単位で同じ音声になりました。乱数の生成器は独自のものなので、同じシードでも PyTorch 版と同じテイクにはなりません。また、バッチの大きさが違うと結果がわずかに変わることがあります。

照合を再現するには、[uv](https://docs.astral.sh/uv/) でスクリプトを実行します。上の検証で使った版のパッケージ（各スクリプトの先頭で固定）が自動で入ります。スクリプトが基準のファイルを書き出し、エンジンが自分の結果と比べます。`--check-tokens` と `--check-case` は、一致すべきものが一致しないとエラーで終了します。

    uv run scripts/hf_tokens.py --model DIR texts.json --out tokens.json   # 文字列のリスト
    oumu --model DIR --check-tokens tokens.json

    uv run scripts/hf_vae.py --model DIR --wav voice.wav --out vae.safetensors
    oumu --model DIR --check-vae vae.safetensors --wav voice.wav

    uv run scripts/hf_case.py --model DIR --ref voice.wav --prompt voice.wav \
        --prompt-text "..." --text "..." --seed 7 --out case.safetensors
    oumu --model DIR --check-case case.safetensors --ref voice.wav --prompt voice.wav --prompt-text "..." --text "..."
    oumu --model DIR --case case.safetensors     # デコードの途中経過を比べる

開発用に、`--case FILE` に `--batch N --noise random` を付けると、N テイクのプレフィルとデコードの時間を測れます（WAV は書きません）。同じく `--gemm-bench`／`--bw` を付けると、手元の GPU で行列積カーネルとメモリ帯域を測ります。

## 制限

- VoxCPM2 専用です（他のチェックポイントは断ります）。LoRA、ストリーミング、録音のノイズ除去、テキストの正規化、長すぎるテイクの自動やり直しはありません。複数のシードで作って選んでください。
- 入力は WAV だけです（`ffmpeg -i in.mp3 out.wav`）。
- ライブラリではなくコマンドラインのツールです。GPU のバッファはプロセスの終了時に解放します。
- Linux と NVIDIA GPU のみです。

## 責任ある利用

複製してよいのは、自分の声か、本人が同意した声だけです。用途も本人が同意した範囲に限ってください。合成音声を、なりすまし、聞き手をだますこと、声による認証の突破に使わないでください。本物の録音と受け取られるおそれがある場面では、合成音声だと明示してください。肖像、同意、表示に関する法律は国によって違います。このエンジンは電子透かしを入れません。

## セキュリティ

ネットワークには接続しません。読むのはモデルのディレクトリ、録音、テキストで、書くのは WAV ファイルだけです。モデルファイル、WAV、JSON は検証付きの読み込み処理を通します（オフセット、テンソルの形、トークン ID、入れ子の深さを確認）。`audiovae.pth` の pickle は許可リスト方式で解釈し、作れるものの大きさにも上限を設けているので、チェックポイントの中のコードは実行されません。ただし、悪意のあるチェックポイントで出力がおかしくなることはありえるので、信頼できる配布元のモデルを使ってください。

## ライセンス

[Apache License 2.0](LICENSE-APACHE) と [MIT](LICENSE-MIT) のどちらかを選べます。モデルの重みは含みません（VoxCPM2 は OpenBMB が Apache-2.0 で公開）。

oumu は OpenBMB の VoxCPM2（[VoxCPM](https://github.com/OpenBMB/VoxCPM)、Apache-2.0）を独自に実装したもので、手順は本家に一つずつ合わせています。Hugging Face `transformers`（トークナイザの挙動）と cudarc の成果も使っています。VoxCPM と `transformers` に由来する部分は [NOTICE](NOTICE) にまとめています。
