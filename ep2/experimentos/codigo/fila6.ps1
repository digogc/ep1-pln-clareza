# Sexta fila (entra quando o BERT de 128 tokens em 5 dobras da fila 5 termina). Retomavel como as anteriores.
#   1. BERT com 256 tokens em 5 dobras (pedido do usuario em 30/10), a partir do mesmo modelo escolhido pelo
#      e15_decidir_tapt.py para o de 128 tokens;
#   2. dobras que faltam do BERT ordinal e do BERT "extremos" (prioridade baixa).
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
$res = Join-Path $PSScriptRoot "..\resultados"
$tapt = Join-Path $HOME "ep2_cache\bertimbau_tapt"
function Log($m) { "$m $(Get-Date)" | Out-File -Append -Encoding utf8 (Join-Path $res "fila6_eventos.txt") }

Log "inicio"
if (Test-Path (Join-Path $res "usar_tapt.txt")) {
    python e02_bert.py --nome tapt256_5d --modelo $tapt --max-len 256 --objetivo ce --threads 8 --dobras 5 --prever-cada-epoca *>> (Join-Path $res "e02_tapt256_5d_log.txt")
} else {
    python e02_bert.py --nome bert256_5d --max-len 256 --objetivo ce --threads 8 --dobras 5 --prever-cada-epoca *>> (Join-Path $res "e02_bert256_5d_log.txt")
}
Log "BERT de 256 tokens em 5 dobras pronto"
python e02_bert.py --nome ordinal --objetivo ordinal --threads 8 --dobras 2 --prever-cada-epoca *>> (Join-Path $res "e02_ordinal_log.txt")
python e02_bert.py --nome extremos --objetivo extremos --threads 8 --dobras 2 --prever-cada-epoca *>> (Join-Path $res "e02_extremos_log.txt")
Log "fila 6 concluida"
