param(
    [ValidateSet('base', 'stack', 'yahoo', 'linkedin', 'quora', 'reddit')]
    [string]$Model,
    [ValidateRange(1024, 8192)]
    [int]$Context = 4096,
    [ValidateRange(1024, 65535)]
    [int]$Port = 8080,
    [ValidateSet('0.4', '0.7', '0.95')]
    [string]$Temperature
)

$ErrorActionPreference = 'Stop'
$config = Get-Content -LiteralPath "$PSScriptRoot/config.json" -Raw | ConvertFrom-Json
$baseFile = 'Qwen4B-Base-IQ4_XS.gguf'
if (!(Test-Path -LiteralPath "$PSScriptRoot/artifacts/gguf/$baseFile")) {
    $baseFile = 'Qwen4B-Base-Q4_K_M.gguf'
}
$files = [ordered]@{base = $baseFile}
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
$sampling = @()
if ($Temperature) {
    $sampling = @('--temp', $Temperature, '--top-k', '20', '--top-p', '0.8', '--min-p', '0',
                  '--repeat-penalty', '1.12', '--repeat-last-n', '128')
}
& $server -m $path --alias $Model --host 127.0.0.1 --port $Port --jinja -ngl 99 -c $Context --parallel 1 @sampling
exit $LASTEXITCODE
