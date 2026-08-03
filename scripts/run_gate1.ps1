# Run Gate 1 end to end: checkpoint -> core RTL -> test vectors -> xsim -> pass/fail.
#
# Gate 1 is the only correctness signal in this project (CLAUDE.md), so it must be trivial to
# re-run and must rebuild everything it checks. It deliberately regenerates the core and the
# vectors from the checkpoint each time rather than trusting whatever is on disk -- a stale
# dwn_core.v passing against stale vectors would be worse than no check at all.
#
# Usage:
#   .\scripts\run_gate1.ps1
#   .\scripts\run_gate1.ps1 -Checkpoint training\artifacts\<other>_checkpoint.pt

param(
    [string]$Checkpoint = "training\artifacts\dwn_jsc_t200_distributive_50_l_b100_checkpoint.pt",
    [string]$VivadoBin  = "C:\AMDDesignTools\2025.2\Vivado\bin"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$work = Join-Path $repo "build\gate1"

if (-not (Test-Path (Join-Path $repo ".venv\Scripts\python.exe"))) {
    throw "No .venv. Run: py -3.12 -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt"
}
if (-not (Test-Path $VivadoBin)) {
    throw "Vivado not found at $VivadoBin. Pass -VivadoBin <path> if it moved."
}

$python = Join-Path $repo ".venv\Scripts\python.exe"
$ckpt   = Join-Path $repo $Checkpoint

Write-Host "`n=== 1/3  emit core RTL from checkpoint ===" -ForegroundColor Cyan
& $python (Join-Path $repo "exporter\emit_core.py") $ckpt
if ($LASTEXITCODE -ne 0) { throw "emit_core.py failed" }

Write-Host "`n=== 2/3  generate test vectors ===" -ForegroundColor Cyan
& $python (Join-Path $repo "tb\gen_vectors.py") $ckpt
if ($LASTEXITCODE -ne 0) { throw "gen_vectors.py failed" }

Write-Host "`n=== 3/3  simulate (xsim) ===" -ForegroundColor Cyan
$env:PATH = "$VivadoBin;$env:PATH"
Push-Location $work
try {
    # xsim writes xsim.dir/, logs and .pb files into the working directory, which is why this
    # runs inside build/ -- nothing outside build/ should accumulate generated artifacts.
    & xvlog.bat -i $work `
        (Join-Path $repo "rtl\lut_node.v") `
        (Join-Path $repo "rtl\popcount.v") `
        (Join-Path $repo "rtl\argmax.v") `
        (Join-Path $repo "rtl\gen\dwn_core.v") `
        (Join-Path $repo "tb\dwn_core_tb.v") | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "xvlog failed" }

    & xelab.bat dwn_core_tb -s gate1_sim -debug off | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "xelab failed" }

    $out = & xsim.bat gate1_sim -runall
    $out | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "xsim failed" }

    # The testbench reports its own verdict; $finish always exits 0, so the pass/fail has to
    # be read out of the output rather than the exit code.
    if ($out -match "RESULT\s+:\s+PASS") {
        Write-Host "`nGATE 1 PASSED" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "`nGATE 1 FAILED" -ForegroundColor Red
        exit 1
    }
}
finally {
    Pop-Location
}
