$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (!(Test-Path '.analyst-venv\Scripts\python.exe')) { py -3 -m venv .analyst-venv }
& .\.analyst-venv\Scripts\python.exe -m pip install -r analyst_service\requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
$tokenFile = Join-Path (Get-Location) '.analyst-service-token'
if (!(Test-Path $tokenFile)) {
  $bytes = New-Object byte[] 32
  $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
  $rng.GetBytes($bytes)
  $rng.Dispose()
  [System.IO.File]::WriteAllText($tokenFile, [Convert]::ToBase64String($bytes))
}
$env:ANALYST_SERVICE_TOKEN = (Get-Content -Raw $tokenFile).Trim()
$env:ANALYST_DATA_DIR = Join-Path (Get-Location) 'analyst-data'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
Write-Host 'Starting TekGPT Python service at http://127.0.0.1:8003'
Write-Host 'The service token is saved in .analyst-service-token. Keep it private; do not paste it in chat.'
& .\.analyst-venv\Scripts\python.exe -m uvicorn analyst_service.app:app --host 127.0.0.1 --port 8003
