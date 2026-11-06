# Quinta fila (30/10, substitui a 4 depois do TAPT). Retomavel como a 4: rodar de novo continua de onde parou.
# Mudancas em relacao a fila 4:
#   - o Qwen3 ja foi feito na GPU (sem ganho no stacking), entao sai da fila;
#   - o BERT de 5 dobras usa 128 tokens (~19 h) em vez de 256 (~40 h): na metade 0, o BERT de 256 tokens nao
#     melhorou o stacking;
#   - a escolha entre TAPT e BERTimbau original e feita pelo e15_decidir_tapt.py, com regra fixa;
#   - as previsoes vao para a GPU (e12_previsor_gpu.py) quando o previsor esta ativo.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
$res = Join-Path $PSScriptRoot "..\resultados"
$tapt = Join-Path $HOME "ep2_cache\bertimbau_tapt"
function Log($m) { "$m $(Get-Date)" | Out-File -Append -Encoding utf8 (Join-Path $res "fila5_eventos.txt") }

Log "inicio"
if (-not (Test-Path (Join-Path $tapt "config.json"))) {
    python e09_tapt.py --epocas 2 --threads 8 *>> (Join-Path $res "e09_tapt_log.txt")
}
python e02_bert.py --nome tapt --modelo $tapt --objetivo ce --threads 8 --dobras 2 --so-dobra 0 --prever-cada-epoca *>> (Join-Path $res "e02_tapt_log.txt")
Log "fine-tuning do TAPT na metade 0 pronto"
# as previsoes podem estar na GPU: espera o arquivo (no maximo 2 h)
$fim = (Get-Date).AddHours(2)
while (-not (Test-Path (Join-Path $res "tapt_dobra0.npz")) -and (Get-Date) -lt $fim) { Start-Sleep -Seconds 30 }
python e15_decidir_tapt.py *>> (Join-Path $res "e15_decidir_tapt_saida.txt")
Log "decisao do TAPT gravada"
if (Test-Path (Join-Path $res "usar_tapt.txt")) {
    python e02_bert.py --nome tapt5d --modelo $tapt --objetivo ce --threads 8 --dobras 5 --prever-cada-epoca *>> (Join-Path $res "e02_tapt5d_log.txt")
} else {
    python e02_bert.py --nome bert5d --objetivo ce --threads 8 --dobras 5 --prever-cada-epoca *>> (Join-Path $res "e02_bert5d_log.txt")
}
Log "BERT de 5 dobras pronto"
python e02_bert.py --nome ordinal --objetivo ordinal --threads 8 --dobras 2 --prever-cada-epoca *>> (Join-Path $res "e02_ordinal_log.txt")
python e02_bert.py --nome extremos --objetivo extremos --threads 8 --dobras 2 --prever-cada-epoca *>> (Join-Path $res "e02_extremos_log.txt")
Log "fila 5 concluida"
