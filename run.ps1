<#
.SYNOPSIS
    Build and run the whole demo: simulate, train, then serve scorer + gateway +
    mock institutions + console.

.DESCRIPTION
    Four processes plus a UI is more surface than a single script, so this exists
    to make a cold start on stage boring. Every service has a health endpoint and
    the script waits for each before starting the next.

.EXAMPLE
    .\run.ps1 -Scenario sloppy            # serve an already-built scenario
    .\run.ps1 -Scenario moderate -Rebuild # regenerate data and retrain first
    .\run.ps1 -Test                       # run both test suites and exit
    .\run.ps1 -Stop                       # stop everything
#>
[CmdletBinding()]
param(
    [string]$Scenario = "sloppy",
    [switch]$Rebuild,
    [switch]$Stop,
    [switch]$Test,
    [int]$ScorerPort = 8000,
    [int]$GatewayPort = 8080,
    [int]$UiPort = 5173
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$pidFile = Join-Path $root ".run-pids.json"

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "  ok  $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "  !!  $msg" -ForegroundColor Yellow }

function Stop-All {
    if (Test-Path $pidFile) {
        $pids = Get-Content $pidFile -Raw | ConvertFrom-Json
        foreach ($p in $pids) {
            # /T kills the whole tree. `npm run dev` is a .cmd shim that spawns
            # the real vite process as a child: stopping only the shim left node
            # holding the port, and the next start then failed on it.
            #
            # Routed through cmd so that cmd, not PowerShell, swallows the
            # stderr. Redirecting a native command's stderr in Windows
            # PowerShell wraps each line in an ErrorRecord, which under
            # `$ErrorActionPreference = "Stop"` is terminating -- so stopping a
            # stack where one process had already exited threw instead of
            # reporting "was not running", which is the exact case this loop
            # exists to handle.
            $null = cmd /c "taskkill /PID $($p.pid) /T /F >nul 2>nul"
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "stopped $($p.name) (pid $($p.pid))"
            } else {
                Write-Warn2 "$($p.name) (pid $($p.pid)) was not running"
            }
        }
        Remove-Item $pidFile -Force
    } else {
        Write-Warn2 "no pid file; nothing tracked to stop"
    }
}

if ($Stop) { Write-Step "Stopping services"; Stop-All; return }

# Both suites run in well under a minute on the tiny scenario, so there is no
# reason not to run them before a demo. The pipeline suite covers the seams
# between modules, which is where every failure so far has actually been.
if ($Test) {
    $python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    $failed = $false
    foreach ($suite in @("tests\test_biohash.py", "tests\test_pipeline.py")) {
        Write-Step "Running $suite"
        & $python $suite
        if ($LASTEXITCODE -ne 0) { $failed = $true }
    }
    if ($failed) { throw "tests failed" }
    Write-Ok "all suites passed"
    return
}

if (-not (Test-Path $python)) {
    throw "no venv at $python. Create it with: python -m venv --system-site-packages .venv"
}

# -- ports ------------------------------------------------------------------
# Check before starting anything. A busy port surfaced sixty seconds later as
# "scorer did not become healthy", which points at the scorer -- when the real
# cause was an unrelated project already listening on 8000. Name the offender
# instead, and say which switch moves us out of its way.
function Assert-PortFree($port, $label, $switch) {
    $conn = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $conn) { return }
    $owner = (Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue)
    $who = if ($owner) { "$($owner.ProcessName) (pid $($owner.Id))" } else { "pid $($conn.OwningProcess)" }
    throw "port $port ($label) is already in use by $who. Stop it, or start on another port with $switch."
}

Assert-PortFree $ScorerPort  "scorer"   "-ScorerPort <n>"
Assert-PortFree $GatewayPort "gateway"  "-GatewayPort <n>"
Assert-PortFree $UiPort      "console"  "-UiPort <n>"
Assert-PortFree 8101         "merchant" "a free 8101"
Assert-PortFree 8102         "issuer"   "a free 8102"

# -- build ------------------------------------------------------------------
if ($Rebuild) {
    Write-Step "Simulating scenario '$Scenario'"
    & $python -m sim.run --scenario $Scenario
    if ($LASTEXITCODE -ne 0) { throw "simulation failed" }

    Write-Step "Verifying the generated data"
    & $python -m sim.verify --data "data/$Scenario"
    if ($LASTEXITCODE -ne 0) { Write-Warn2 "verification reported failures (continuing)" }

    Write-Step "Training models"
    & $python -m detect.models.train --data "data/$Scenario"
    if ($LASTEXITCODE -ne 0) { throw "training failed" }
}

if (-not (Test-Path (Join-Path $root "data\$Scenario\meta.json"))) {
    throw "no data for '$Scenario'. Run with -Rebuild first."
}
if (-not (Test-Path (Join-Path $root "models\$Scenario\behaviour.pkl"))) {
    throw "no trained models for '$Scenario'. Run with -Rebuild first."
}

# -- node dependencies ------------------------------------------------------
foreach ($dir in @("gateway", "ui")) {
    if (-not (Test-Path (Join-Path $root "$dir\node_modules"))) {
        Write-Step "Installing $dir dependencies"
        Push-Location (Join-Path $root $dir); npm install; Pop-Location
    }
}
Write-Step "Building the gateway"
Push-Location (Join-Path $root "gateway"); npx tsc; Pop-Location

# -- start ------------------------------------------------------------------
$started = @()

function Save-Pids($started) {
    $started | ConvertTo-Json | Set-Content $pidFile -Encoding utf8
}

function Wait-Healthy($url, $name, $timeoutSec = 60) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-RestMethod -Uri $url -TimeoutSec 3
            Write-Ok "$name healthy"
            return $r
        } catch { Start-Sleep -Milliseconds 700 }
    }
    throw "$name did not become healthy within ${timeoutSec}s"
}

Write-Step "Starting the scorer (FastAPI) on :$ScorerPort"
$env:FRAUD_DATA = "data/$Scenario"
$scorer = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "$ScorerPort" `
    -WorkingDirectory $root -PassThru -WindowStyle Hidden
$started += @{ name = "scorer"; pid = $scorer.Id }
Save-Pids $started
Wait-Healthy "http://127.0.0.1:$ScorerPort/health" "scorer" | Out-Null

Write-Step "Building the identity graph"
# Off the auth path by design: this is the expensive pass, done once.
$graph = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$ScorerPort/admin/build-graph" -TimeoutSec 900
Write-Ok "graph: $($graph.graph)"
Write-Ok "communities: $($graph.communities)"

Write-Step "Starting the gateway (Fastify) on :$GatewayPort"
# Set the variables on this process and let the child inherit them.
# `Start-Process -Environment` is PowerShell 7 only, and Windows 11 ships
# Windows PowerShell 5.1 as `powershell.exe` -- on a machine without pwsh this
# script died here, which is the worst possible place for it to die.
$env:SCORER_URL = "http://127.0.0.1:$ScorerPort"
$env:GATEWAY_PORT = "$GatewayPort"
$gateway = Start-Process -FilePath "node" -ArgumentList "dist/server.js" `
    -WorkingDirectory (Join-Path $root "gateway") -PassThru -WindowStyle Hidden
$started += @{ name = "gateway"; pid = $gateway.Id }
Save-Pids $started
Wait-Healthy "http://127.0.0.1:$GatewayPort/health" "gateway" | Out-Null

Write-Step "Starting mock merchant and issuer services"
foreach ($svc in @(
    @{ role = "merchant"; inst = "inst_00"; port = 8101 },
    @{ role = "issuer";   inst = "inst_01"; port = 8102 }
)) {
    $env:GATEWAY_URL = "http://127.0.0.1:$GatewayPort"
    $p = Start-Process -FilePath "node" `
        -ArgumentList "dist/institution.js", "--role", $svc.role, "--institution", $svc.inst, "--port", "$($svc.port)" `
        -WorkingDirectory (Join-Path $root "gateway") -PassThru -WindowStyle Hidden
    $started += @{ name = "$($svc.role):$($svc.inst)"; pid = $p.Id }
    Save-Pids $started
    Wait-Healthy "http://127.0.0.1:$($svc.port)/health" "$($svc.role) $($svc.inst)" | Out-Null
}

Write-Step "Starting the console on :$UiPort"
# `npm` on Windows is a .cmd shim, and Start-Process will not resolve the
# bare name -- it fails with "cannot find all the information required".
# Resolve the shim explicitly.
$npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
if (-not $npm) { $npm = (Get-Command npm -ErrorAction Stop).Source }
$ui = Start-Process -FilePath $npm -ArgumentList "run", "dev", "--", "--port", "$UiPort" `
    -WorkingDirectory (Join-Path $root "ui") -PassThru -WindowStyle Hidden
$started += @{ name = "ui"; pid = $ui.Id }
Save-Pids $started

# Written unconditionally, so `-Stop` can still clean up if a later start
# failed: without this, a throw partway through left orphaned node and python
# processes holding the ports and nothing recorded to kill them.
$started | ConvertTo-Json | Set-Content $pidFile -Encoding utf8

Write-Host ""
Write-Host "  console   http://127.0.0.1:$UiPort" -ForegroundColor White
Write-Host "  gateway   http://127.0.0.1:$GatewayPort/health"
Write-Host "  scorer    http://127.0.0.1:$ScorerPort/docs"
Write-Host "  merchant  http://127.0.0.1:8101/health   (sees only inst_00)"
Write-Host "  issuer    http://127.0.0.1:8102/health   (sees only inst_01)"
Write-Host ""
Write-Host "  stop everything with:  .\run.ps1 -Stop" -ForegroundColor DarkGray
