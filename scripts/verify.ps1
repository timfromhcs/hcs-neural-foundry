<#
.SYNOPSIS
    Runs every local gate for HCS Neural Foundry and prints a compact report.

.DESCRIPTION
    Gates:
      1. Python lint (ruff) and tests (pytest)
      2. Web typecheck, unit tests and production build
      3. Locale parity is covered by the web tests; docs presence is checked here
      4. End-to-end smoke: create -> train -> evaluate -> patch -> export in a throwaway workspace

    Anything that fails prints a non-zero exit code. Nothing is published or pushed.

.EXAMPLE
    .\scripts\verify.ps1
    .\scripts\verify.ps1 -SkipWeb
#>

param(
    [switch]$SkipWeb,
    [switch]$SkipSmoke,
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$failures = New-Object System.Collections.Generic.List[string]
$timings = New-Object System.Collections.Generic.List[object]

function Invoke-Gate {
    param([string]$Name, [scriptblock]$Action)
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    $start = Get-Date
    try {
        & $Action
        if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
        $elapsed = (Get-Date) - $start
        $timings.Add([pscustomobject]@{ Gate = $Name; Seconds = [math]::Round($elapsed.TotalSeconds, 2); Status = "pass" })
        Write-Host "PASS  $Name ($([math]::Round($elapsed.TotalSeconds, 2)) s)" -ForegroundColor Green
    }
    catch {
        $elapsed = (Get-Date) - $start
        $timings.Add([pscustomobject]@{ Gate = $Name; Seconds = [math]::Round($elapsed.TotalSeconds, 2); Status = "FAIL" })
        Write-Host "FAIL  $Name : $_" -ForegroundColor Red
        $failures.Add("$Name : $_")
    }
}

if (-not (Test-Path $Python)) {
    Write-Host "Python environment not found at $Python" -ForegroundColor Red
    Write-Host 'Create it with: uv venv .venv --python 3.12; uv pip install --python .venv\Scripts\python.exe -e ".[torch,export,mcp,dev]"'
    exit 2
}

Invoke-Gate "python lint" { & $Python -m ruff check python }
Invoke-Gate "python tests" { & $Python -m pytest python/tests -q }

if (-not $SkipWeb) {
    if (Test-Path "apps\web\node_modules") {
        Push-Location apps\web
        try {
            Invoke-Gate "web typecheck" { & npm run typecheck }
            Invoke-Gate "web tests" { & npm test }
            Invoke-Gate "web build" { & npm run build }
        }
        finally { Pop-Location }
    }
    else {
        Write-Host ""
        Write-Host "SKIP  web gates (run 'npm install' in apps\web)" -ForegroundColor Yellow
    }
}

Invoke-Gate "docs presence" {
    $required = @(
        "README.md", "plan.md", "DECISIONS.md", "LICENSE",
        "docs\RESEARCH.md", "docs\ARCHITECTURE.md", "docs\TESTING.md",
        "docs\SECURITY.md", "docs\STATUS.md", "docs\PERFORMANCE.md",
        "docs\acceptance.json"
    )
    $missing = $required | Where-Object { -not (Test-Path $_) }
    if ($missing) { throw "missing documents: $($missing -join ', ')" }
    $lessons = Get-ChildItem "docs\tutorials" -Filter "*.md" -ErrorAction SilentlyContinue
    if ($lessons.Count -lt 20) { throw "expected 20 tutorial lessons, found $($lessons.Count)" }
}

if (-not $SkipSmoke) {
    Invoke-Gate "end-to-end smoke" {
        $smoke = Join-Path $env:TEMP ("foundry-verify-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
        $previous = $env:HCS_FOUNDRY_HOME
        $env:HCS_FOUNDRY_HOME = $smoke
        try {
            & $Python -m hcs_foundry model create --name "Smoke LM" --params 120k
            if ($LASTEXITCODE -ne 0) { throw "model create failed" }
            $modelId = (& $Python -m hcs_foundry model list --json | ConvertFrom-Json)[0].id
            & $Python -m hcs_foundry train --model $modelId --steps 12 --batch-size 8 --block-size 16 --eval-every 6 --checkpoint-every 12 --max-seconds 120
            if ($LASTEXITCODE -ne 0) { throw "training failed" }
            & $Python -m hcs_foundry eval --model $modelId --suite text-smoke --batches 2
            if ($LASTEXITCODE -ne 0) { throw "evaluation failed" }
            & $Python -m hcs_foundry weights list --model $modelId --filter q_proj --json | Out-Null
            # Native argument quoting: PowerShell needs the inner quotes escaped to reach the CLI intact.
            & $Python -m hcs_foundry weights patch --model $modelId --tensor blocks.0.attn.q_proj.weight --op scale --params '{\"factor\":0.99}' | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "patch failed" }
            & $Python -m hcs_foundry export --model $modelId --format onnx | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "export failed" }
            Write-Host "smoke workflow completed for $modelId"
        }
        finally {
            $env:HCS_FOUNDRY_HOME = $previous
            Remove-Item -Recurse -Force $smoke -ErrorAction SilentlyContinue
        }
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
$timings | Format-Table -AutoSize

if ($failures.Count -gt 0) {
    Write-Host "FAILED gates:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host " - $_" -ForegroundColor Red }
    exit 1
}

Write-Host "All gates passed." -ForegroundColor Green
exit 0
