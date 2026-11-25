$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (!(Test-Path -LiteralPath '.venv-ner/Scripts/python.exe')) {
    python -m venv .venv-ner
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& .venv-ner/Scripts/python.exe -m pip --disable-pip-version-check install --no-cache-dir `
    -r requirements-preprocessing.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .venv-ner/Scripts/python.exe -m pip check
exit $LASTEXITCODE
