# Segunda fila: espera a primeira (ordinal + extremos) terminar, faz o TAPT e o fine-tuning do modelo adaptado.
param([int]$Esperar = 0)
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
$res = Join-Path $PSScriptRoot "..\resultados"
$tapt = Join-Path $PSScriptRoot "..\cache\bertimbau_tapt"
if ($Esperar -gt 0) { Wait-Process -Id $Esperar -ErrorAction SilentlyContinue }
if (-not (Test-Path (Join-Path $tapt "config.json"))) {
    python e09_tapt.py --epocas 2 --threads 8 *> (Join-Path $res "e09_tapt_log.txt")
}
python e02_bert.py --nome tapt --modelo $tapt --objetivo ce --threads 8 --dobras 2 --prever-cada-epoca *> (Join-Path $res "e02_tapt_log.txt")
"fila 2 concluida $(Get-Date)" | Out-File (Join-Path $res "fila_noite2_fim.txt")
