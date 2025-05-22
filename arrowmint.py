# main.py — ArrowMint 4B (INT4) を CPU でリアルタイム対話
from pathlib import Path
import os
from huggingface_hub import snapshot_download
from llama_cpp import Llama

# -------- モデル指定 --------
REPO  = "mmnga/ArrowMint-Gemma3-4B-YUKI-v0.1-gguf"
FNAME = "ArrowMint-Gemma3-4B-YUKI-v0.1-Q4_K_M.gguf"
MODEL = Path("models") / FNAME

# -------- ダウンロード（初回のみ）--------
if not MODEL.exists():                       # ファイルが無ければ DL
    snapshot_download(repo_id=REPO,
                      allow_patterns=FNAME,
                      local_dir="models")

# -------- LLM のロード --------
llm = Llama(
    model_path=str(MODEL),    # GGUF パス
    n_ctx=2048,               # 最大入力トークン数
    n_threads=os.cpu_count(), # 論理 CPU コアを全部使う
    chat_format="gemma"       # Gemma 用テンプレートを明示（省略可）
)

# -------- 対話ループ（トークンをリアルタイム出力） --------
print("🔮 LLM Chat — type 'exit' to quit")
history = []  # 過去メッセージを保存し、文脈を維持

while True:
    user = input("You: ")
    if user.lower() in {"exit", "quit"}:
        break  # ループ終了

    history.append({"role": "user", "content": user})

    assistant_tokens = []
    # stream=True: 1 トークンずつ chunk が届く
    for chunk in llm.create_chat_completion(
        messages=history,
        max_tokens=256,          # max_tokens: 生成する最大トークン数（応答の長さの上限）
        temperature=0.7,         # temperature: 出力のランダム性を制御（低いほど決定的、高いほど多様性が増える）
        top_p=0.9,               # top_p: nucleus sampling の確率質量上位 p を母集合に採用
        stream=True):             # stream: True で逐次チャンクを受け取りながら生成（ストリーム出力）
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta:               # 実際のトークン
            token = delta["content"]
            print(token, end="", flush=True) # 即時表示
            assistant_tokens.append(token)
    print()                                   # 応答終了 → 改行

    answer = "".join(assistant_tokens)        # トークンを連結して全文に
    history.append({"role": "assistant", "content": answer})