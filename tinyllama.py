# main.py — TinyLlama (INT4) を CPU で動かす最短サンプル
from pathlib import Path
import os
from huggingface_hub import snapshot_download
from llama_cpp import Llama

# 初級：1Bモデル。700 MBくらいで早いが賢くはない
REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
FNAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # INT4

# これより性能のいいものとして、llamaやgemmaなどがあるが、それらはhugging faceで登録とリクエストが必要（無料）


MODEL = Path("models") / FNAME

# ① モデルが無ければ自動ダウンロード（初回のみ数分）
if not MODEL.exists():
    snapshot_download(repo_id=REPO, allow_patterns=FNAME, local_dir="models")

# ② モデルをロード（CPU スレッド数は自動）
llm = Llama(model_path=str(MODEL), n_ctx=2048, n_threads=os.cpu_count())

# VSCodeなら、ctrl+/で複数行を一気にコメント化できます
# # ③ 対話ループ（一括表示）
# print("🔮 LLM Chat — type 'exit' to quit")
# while True:
#     user = input("You: ")
#     if user.lower() in {"exit", "quit"}: break

#     prompt = f"### ユーザ:{user}\n### アシスタント:"
#     res = llm(prompt, max_tokens=256, stop=["### ユーザ:", "### アシスタント:"])
#     print("AI:", res["choices"][0]["text"].strip())


# ③ 対話ループ（トークンをリアルタイムに表示）
print("🔮 LLM Chat — type 'exit' to quit")
while True:                                    # 無限ループで対話を継続
    user = input("You: ")                     # ← ユーザ入力を取得 (Python 基本構文)
    if user.lower() in {"exit", "quit"}:      # 小文字化して終了ワード判定
        break                                  # break で while ループを抜ける

    # プロンプトを Llama.cpp 用フォーマットで組み立て
    # 例: "### ユーザ:こんにちは
    prompt = f"### ユーザ:{user}"

    # stream=True にすると生成を 1 トークンずつ受け取れる
    # ストリーミング生成されたチャンクを順次処理するループ
    for chunk in llm.create_completion(
                prompt,                  # prompt: モデルに渡す入力テキスト（質問・指示など）
                max_tokens=256,          # max_tokens: 生成する最大トークン数（応答の長さの上限）
                temperature=0.7,         # temperature: 出力のランダム性を制御（低いほど決定的、高いほど多様性が増える）
                top_p=0.9,               # top_p: nucleus sampling の確率質量上位 p を母集合に採用
                stream=True,             # stream: True で逐次チャンクを受け取りながら生成（ストリーム出力）
                stop=["### ユーザ:", "### アシスタント:"]  # stop: これらの文字列が現れた時点で生成を打ち切る
            ):
        token = chunk["choices"][0]["text"]   # 出力トークン文字列
        print(token, end="", flush=True)       # flush=True でバッファ即時出力
    print()  # 応答が終わったら改行して次の入力へ