# Quarta fila (30/10, depois que o Windows reiniciou sozinho as 06:16 para instalar uma atualizacao).
# Todos os passos sao retomaveis: rodar a fila de novo depois de um reinicio continua de onde parou
# (dobras prontas sao puladas, treinos voltam do ultimo checkpoint em ~/ep2_ckpt, partes de embeddings prontas sao puladas).
#   1. TAPT (pre-treino nos textos do treino) e fine-tuning do modelo adaptado na metade 0, 128 tokens
#      (comparavel com a metade 0 do BERT da v2);
#   2. embeddings do Qwen3-Embedding-0.6B (sondagem interrompida antes) e o bloco no stacking;
#   3. BERT com 256 tokens em 5 dobras (as mesmas do stacking), a partir do modelo adaptado se o arquivo
#      usar_tapt.txt existir em resultados/ (decidido pela comparacao do passo 1), senao do BERTimbau original;
#   4. dobras que faltam do BERT ordinal e do BERT "extremos" (prioridade baixa).
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
$res = Join-Path $PSScriptRoot "..\resultados"
$tapt = Join-Path $HOME "ep2_cache\bertimbau_tapt"
function Log($m) { "$m $(Get-Date)" | Out-File -Append (Join-Path $res "fila4_eventos.txt") }

Log "inicio"
if (-not (Test-Path (Join-Path $tapt "config.json"))) {
    python e09_tapt.py --epocas 2 --threads 8 *>> (Join-Path $res "e09_tapt_log.txt")
}
Log "tapt pronto"
python e02_bert.py --nome tapt --modelo $tapt --objetivo ce --threads 8 --dobras 2 --so-dobra 0 --prever-cada-epoca *>> (Join-Path $res "e02_tapt_log.txt")
Log "tapt metade 0 pronto"
python e04_embeddings.py --threads 8 --lote 8 *>> (Join-Path $res "e04_qwen3_log.txt")
Log "qwen3 pronto"
if (Test-Path (Join-Path $res "usar_tapt.txt")) {
    python e02_bert.py --nome tapt256_5d --modelo $tapt --max-len 256 --objetivo ce --threads 8 --dobras 5 *>> (Join-Path $res "e02_tapt256_5d_log.txt")
} else {
    python e02_bert.py --nome len256_5d --max-len 256 --objetivo ce --threads 8 --dobras 5 *>> (Join-Path $res "e02_len256_5d_log.txt")
}
Log "bert 256 em 5 dobras pronto"
python e02_bert.py --nome ordinal --objetivo ordinal --threads 8 --dobras 2 --prever-cada-epoca *>> (Join-Path $res "e02_ordinal_log.txt")
python e02_bert.py --nome extremos --objetivo extremos --threads 8 --dobras 2 --prever-cada-epoca *>> (Join-Path $res "e02_extremos_log.txt")
Log "fila 4 concluida"
