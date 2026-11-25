$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = '.venv/Scripts/python.exe'

function Invoke-Pipeline {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Pipeline command failed with exit code $LASTEXITCODE" }
}

foreach ($source in @('base', 'stack', 'yahoo', 'linkedin')) {
    Invoke-Pipeline acquire.py $source
}
Invoke-Pipeline prepare.py stack
Invoke-Pipeline prepare.py yahoo --scan-limit 45000
Invoke-Pipeline prepare_extra.py linkedin --allow-smaller
foreach ($source in @('quora', 'reddit')) {
    try {
        Invoke-Pipeline acquire.py $source
        Invoke-Pipeline prepare_extra.py $source
    }
    catch {
        Write-Warning "$source SKIPPED — Suitable corpus unavailable"
    }
}
Invoke-Pipeline study.py
