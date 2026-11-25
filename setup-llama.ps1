$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$tag = 'b12728'
$dest = 'artifacts/llama'
New-Item -ItemType Directory -Force "$dest/bin" | Out-Null
$checksums = [ordered]@{
    'llama-b12728-bin-win-cuda-12.4-x64.zip' = '85b6e332ed7b40fd7a240ac7f32c4f1a29e9ddf6b52853925a215fda23ef24b3'
    'cudart-llama-bin-win-cuda-12.4-x64.zip' = '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'
}
foreach ($asset in $checksums.Keys) {
    $path = "$dest/$asset"
    if (!(Test-Path -LiteralPath $path)) {
        curl.exe --fail --silent --show-error --location --retry 3 --continue-at - `
            --output "$path.part" "https://github.com/ggml-org/llama.cpp/releases/download/$tag/$asset"
        if ($LASTEXITCODE -ne 0) { throw "Download failed for $asset" }
        Move-Item -LiteralPath "$path.part" -Destination $path
    }
    if ((Get-FileHash -LiteralPath $path).Hash -ne $checksums[$asset]) { throw "Checksum mismatch for $asset" }
    Expand-Archive -LiteralPath $path -DestinationPath "$dest/bin" -Force
}
if (!(Test-Path -LiteralPath "$dest/llama.cpp-$tag/convert_hf_to_gguf.py")) {
    curl.exe --fail --silent --show-error --location --retry 3 --output "$dest/source.zip" `
        'https://codeload.github.com/ggml-org/llama.cpp/zip/ea7229481'
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp source download failed' }
    Expand-Archive -LiteralPath "$dest/source.zip" -DestinationPath $dest -Force
    $source = Get-ChildItem -LiteralPath $dest -Directory | Where-Object Name -match '^llama.cpp-ea7229481'
    Move-Item -LiteralPath $source.FullName -Destination "$dest/llama.cpp-$tag"
}
& "$dest/bin/llama-server.exe" --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
