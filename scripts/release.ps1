<#
.SYNOPSIS
    Runs every release gate for HCS Neural Foundry, fail-fast, and prints one table.

.DESCRIPTION
    This is the only command that should precede a tag. It runs, in order:
      1. python lint (ruff)
      2. python tests (pytest)
      3. web typecheck, unit tests and production build
      4. release audit (version consistency, docs, acceptance matrix, secrets, encoding,
         demo manifest, MCP SDK pin, built-site externals) in --strict mode
      5. encoding scan and dependency inventory for the record

    It publishes nothing. A non-zero exit code means: do not tag.

.EXAMPLE
    .\scripts\release.ps1
    .\scripts\release.ps1 -SkipWeb
#>

param(
    [switch]$SkipWeb,
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$results = New-Object System.Collections.Generic.List[object]

function Invoke-Gate {
    param([string]$Name, [scriptblock]$Action)
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    $start = Get-Date

    # Native tools writing to stderr (vite does) must not be treated as failures, and
    # $LASTEXITCODE must not leak in from the previous gate. See scripts/verify.ps1.
    $global:LASTEXITCODE = 0
    $status = 0
    $failure = $null
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Action
        $status = $LASTEXITCODE
        if ($null -eq $status) { $status = 0 }
    }
    catch {
        $status = 1
        $failure = $_.Exception.Message
    }
    finally {
        $ErrorActionPreference = $previous
    }

    $elapsed = (Get-Date) - $start
    if ($status -ne 0) {
        $reason = if ($failure) { $failure } else { "exit code $status" }
        $results.Add([pscustomobject]@{ Gate = $Name; Seconds = [math]::Round($elapsed.TotalSeconds, 1); Status = "FAIL" })
        Write-Host "FAIL  $Name : $reason" -ForegroundColor Red
        Write-Host ""
        Write-Host "Release blocked. Fix the failing gate; do not tag." -ForegroundColor Red
        $results | Format-Table -AutoSize
        exit 1
    }
    $results.Add([pscustomobject]@{ Gate = $Name; Seconds = [math]::Round($elapsed.TotalSeconds, 1); Status = "pass" })
    Write-Host "PASS  $Name ($([math]::Round($elapsed.TotalSeconds, 1)) s)" -ForegroundColor Green
}

if (-not (Test-Path $Python)) {
    Write-Host "Python environment not found at $Python" -ForegroundColor Red
    Write-Host 'Create it with: uv venv .venv --python 3.12; uv pip install --python .venv\Scripts\python.exe -e ".[torch,export,mcp,dev]"'
    exit 2
}

Invoke-Gate "python lint" { & $Python -m ruff check python }
Invoke-Gate "python tests" { & $Python -m pytest python/tests -q }

if (-not $SkipWeb) {
    if (-not (Test-Path "apps\web\node_modules")) {
        Write-Host "apps\web\node_modules missing; run 'npm ci' in apps/web" -ForegroundColor Red
        exit 2
    }
    Push-Location apps\web
    try {
        Invoke-Gate "web typecheck" { & npm run typecheck }
        Invoke-Gate "web tests" { & npm test }
        Invoke-Gate "web build" { & npm run build }
    }
    finally { Pop-Location }
}

Invoke-Gate "release audit (strict)" { & $Python scripts\verify_release.py --strict }
Invoke-Gate "encoding scan" { & $Python scripts\scan_encoding.py }
Invoke-Gate "dependency inventory" { & $Python scripts\dependency_inventory.py | Out-Null }

Write-Host ""
Write-Host "=== Release gates ===" -ForegroundColor Cyan
$results | Format-Table -AutoSize
Write-Host "All release gates passed. Nothing was published; tag only after a green CI run." -ForegroundColor Green
exit 0
