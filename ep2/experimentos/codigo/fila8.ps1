# Oitava fila (03/11, pedido do usuario: hipotese do professor de que mais dobras ajudam). Retomavel como as anteriores.
#   1. BERT com 256 tokens em 10 dobras (cada modelo treina com 90% dos dados; no teste, media de 10 modelos).
#      Com os lotes agrupados por tamanho (--agrupar): sem isso seriam ~87 h (fim no sabado 07/11, perto demais do
#      prazo); com ele, ~55 h. Diferenca para o bert256_5d: 10 dobras + agrupamento (este so muda a eficiencia).
#   2. dobras que faltam do BERT ordinal e do BERT "extremos" (prioridade baixa, retomam do checkpoint).
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
$res = Join-Path $PSScriptRoot "..\resultados"
function Log($m) { "$m $(Get-Date)" | Out-File -Append -Encoding utf8 (Join-Path $res "fila8_eventos.txt") }

Log "inicio"
python e02_bert.py --nome bert256_10d --max-len 256 --agrupar --objetivo ce --threads 8 --dobras 10 *>> (Join-Path $res "e02_bert256_10d_log.txt")
Log "BERT de 256 tokens em 10 dobras pronto"
python e02_bert.py --nome ordinal --objetivo ordinal --threads 8 --dobras 2 --prever-cada-epoca *>> (Join-Path $res "e02_ordinal_log.txt")
python e02_bert.py --nome extremos --objetivo extremos --threads 8 --dobras 2 --prever-cada-epoca *>> (Join-Path $res "e02_extremos_log.txt")
Log "fila 8 concluida"
