# Setima fila (02/11, substitui a 6 depois do BERT de 256 tokens). Retomavel como as anteriores.
#   1. BERT com 512 tokens em 5 dobras: o de 256 melhorou o stacking em todas as dobras (47,29 -> 47,74) e 36% dos
#      textos passam de 256 tokens. Com os lotes agrupados por tamanho (--agrupar), custa o mesmo que o de 256
#      (bench: 1,27 textos/s contra 1,16). Sem --prever-cada-epoca: as previsoes da epoca 1 nunca foram melhores.
#   2. dobras que faltam do BERT ordinal e do BERT "extremos" (prioridade baixa, retomam do checkpoint).
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
$res = Join-Path $PSScriptRoot "..\resultados"
function Log($m) { "$m $(Get-Date)" | Out-File -Append -Encoding utf8 (Join-Path $res "fila7_eventos.txt") }

Log "inicio"
python e02_bert.py --nome bert512_5d --max-len 512 --agrupar --objetivo ce --threads 8 --dobras 5 *>> (Join-Path $res "e02_bert512_5d_log.txt")
Log "BERT de 512 tokens em 5 dobras pronto"
python e02_bert.py --nome ordinal --objetivo ordinal --threads 8 --dobras 2 --prever-cada-epoca *>> (Join-Path $res "e02_ordinal_log.txt")
python e02_bert.py --nome extremos --objetivo extremos --threads 8 --dobras 2 --prever-cada-epoca *>> (Join-Path $res "e02_extremos_log.txt")
Log "fila 7 concluida"
