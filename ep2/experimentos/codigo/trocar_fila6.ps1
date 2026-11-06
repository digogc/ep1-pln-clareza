# Troca a fila 5 pela 6 quando o BERT de 128 tokens em 5 dobras terminar (a fila 5 acabou de comecar o ordinal,
# que fica para depois do BERT de 256 tokens; perde-se ~1 min).
$res = Join-Path $PSScriptRoot "..\resultados"
$ev = Join-Path $res "fila5_eventos.txt"
while (-not ((Test-Path $ev) -and (Get-Content $ev | Select-String "BERT de 5 dobras pronto"))) { Start-Sleep -Seconds 20 }
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -like "*fila5.ps1*" } | ForEach-Object {
    $pai = $_.ProcessId
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$pai" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Stop-Process -Id $pai -Force
}
Start-Sleep -Seconds 5
Start-Process powershell.exe -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSScriptRoot\fila6.ps1`"" -WindowStyle Hidden
"fila 5 trocada pela 6 em $(Get-Date)" | Out-File -Append -Encoding utf8 (Join-Path $res "fila6_eventos.txt")
