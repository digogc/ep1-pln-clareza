# Terceira fila (substitui a segunda): quando o BERT ordinal terminar, cancela o BERT "extremos" da primeira fila
# (redundante: escore com correlacao 0,90 com o da v2) e roda, em ordem:
#   1. BERT com 256 tokens (63% das respostas passam de 128 tokens e tinham o meio cortado);
#   2. TAPT (pre-treino nos textos do treino) e o fine-tuning do modelo adaptado.
param([int]$Ordinal = 0, [int]$Fila1 = 0)
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
$res = Join-Path $PSScriptRoot "..\resultados"
$tapt = Join-Path $PSScriptRoot "..\cache\bertimbau_tapt"
if ($Ordinal -gt 0) { Wait-Process -Id $Ordinal -ErrorAction SilentlyContinue }
$fim = (Get-Date).AddMinutes(10)
while ((Get-Date) -lt $fim -and (Get-Process -Id $Fila1 -ErrorAction SilentlyContinue)) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*--nome extremos*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "extremos cancelado $(Get-Date)" | Out-File -Append (Join-Path $res "fila_noite3_eventos.txt") }
    Start-Sleep -Seconds 3
}
python e02_bert.py --nome len256 --max-len 256 --objetivo ce --threads 8 --dobras 2 --prever-cada-epoca *> (Join-Path $res "e02_len256_log.txt")
if (-not (Test-Path (Join-Path $tapt "config.json"))) {
    python e09_tapt.py --epocas 2 --threads 8 *> (Join-Path $res "e09_tapt_log.txt")
}
python e02_bert.py --nome tapt --modelo $tapt --objetivo ce --threads 8 --dobras 2 --prever-cada-epoca *> (Join-Path $res "e02_tapt_log.txt")
"fila 3 concluida $(Get-Date)" | Out-File (Join-Path $res "fila_noite3_fim.txt")
