# Troca a fila 4 pela 5 assim que o TAPT terminar (a fila 4 acabou de comecar o fine-tuning, perde-se ~1 min).
param([int]$Fila4 = 0)
$res = Join-Path $PSScriptRoot "..\resultados"
$ev = Join-Path $res "fila4_eventos.txt"
while (-not ((Test-Path $ev) -and (Get-Content $ev | Select-String "tapt pronto"))) { Start-Sleep -Seconds 20 }
Get-CimInstance Win32_Process -Filter "ParentProcessId=$Fila4" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Stop-Process -Id $Fila4 -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5
Start-Process powershell.exe -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSScriptRoot\fila5.ps1`"" -WindowStyle Hidden
"fila 4 trocada pela 5 em $(Get-Date)" | Out-File -Append -Encoding utf8 (Join-Path $res "fila5_eventos.txt")
