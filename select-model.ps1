param(
    [ValidateSet('base', 'stack', 'yahoo', 'linkedin', 'quora', 'reddit')]
    [string]$Model,
    [ValidateRange(1024, 8192)]
    [int]$Context = 4096,
    [ValidateRange(1024, 65535)]
    [int]$Port = 8080
)

$ErrorActionPreference = 'Stop'
$config = Get-Content -LiteralPath "$PSScriptRoot/config.json" -Raw | ConvertFrom-Json
$files = [ordered]@{base = 'Qwen4B-Base-Q4_K_M.gguf'}
foreach ($name in @('stack', 'yahoo', 'linkedin', 'quora', 'reddit')) {
    $files[$name] = 'Qwen4B-' + $config.$name.name + '-IQ4_XS.gguf'
}
if (!$Model) {
    foreach ($name in $files.Keys) {
        [pscustomobject]@{Model = $name; Ready = Test-Path -LiteralPath "$PSScriptRoot/artifacts/gguf/$($files[$name])"}
    }
    exit
}
$path = "$PSScriptRoot/artifacts/gguf/$($files[$Model])"
if (!(Test-Path -LiteralPath $path)) { throw "Export for $Model is unavailable." }
$server = "$PSScriptRoot/artifacts/llama/bin/llama-server.exe"
& $server -m $path --alias $Model --host 127.0.0.1 --port $Port --jinja -ngl 99 -c $Context --parallel 1
exit $LASTEXITCODE
