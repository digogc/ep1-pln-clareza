# Fila de treinos do BERT: espera o LoRA (processo atual) terminar e roda as variantes de objetivo em sequencia.
param([int]$Esperar = 0)
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
$res = Join-Path $PSScriptRoot "..\resultados"
if ($Esperar -gt 0) { Wait-Process -Id $Esperar -ErrorAction SilentlyContinue }
python e02_bert.py --nome ordinal --objetivo ordinal --threads 8 --dobras 2 --prever-cada-epoca *> (Join-Path $res "e02_ordinal_log.txt")
python e02_bert.py --nome extremos --objetivo extremos --threads 8 --dobras 2 --prever-cada-epoca *> (Join-Path $res "e02_extremos_log.txt")
"fila concluida $(Get-Date)" | Out-File (Join-Path $res "fila_noite_fim.txt")
