# run_experiments.ps1
# Batch testing for ARAIP Part 2. Runs the same random block layouts with:
#   v1_random      - my Part 1 controller (random wander, RGB rules)
#   v2_python_nopf - Part 2 controller, pure Python HSV, pre-filter off
#   v2_python      - Part 2 controller, pure Python HSV + pre-filter
#   v2_opencv      - Part 2 controller, OpenCV HSV (Session 9 method) + pre-filter
# Results are appended to results/runs.csv by the experiment_manager supervisor.
# Usage:  powershell -ExecutionPolicy Bypass -File run_experiments.ps1 [-Runs 10]

param([int]$Runs = 10)

$webots = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
$worlds = Join-Path $PSScriptRoot "worlds"
$base   = Join-Path $worlds "block_search_part2.wbt"
$v1     = Join-Path $worlds "_batch_v1.wbt"

# temporary world that uses the Part 1 controller instead
(Get-Content $base -Raw).Replace('controller "epuck_block_search_v2"', 'controller "epuck_block_search"') |
    Set-Content $v1 -NoNewline -Encoding ascii

$configs = @(
    @{ Label = "v1_random";      World = $v1;   Prefilter = "1"; Vision = "python" },
    @{ Label = "v2_python_nopf"; World = $base; Prefilter = "0"; Vision = "python" },
    @{ Label = "v2_python";      World = $base; Prefilter = "1"; Vision = "python" },
    @{ Label = "v2_opencv";      World = $base; Prefilter = "1"; Vision = "opencv" }
)

$env:ARAIP_BATCH = "1"
$env:ARAIP_TIMEOUT = "180"
foreach ($cfg in $configs) {
    foreach ($seed in 1..$Runs) {
        $env:ARAIP_SEED = "$seed"
        $env:ARAIP_LABEL = $cfg.Label
        $env:ARAIP_PREFILTER = $cfg.Prefilter
        $env:ARAIP_VISION = $cfg.Vision
        Write-Host "Running $($cfg.Label) seed $seed ..."
        $p = Start-Process -FilePath $webots -PassThru -ArgumentList `
            "--mode=fast", "--no-rendering", "--minimize", "--batch", "`"$($cfg.World)`""
        if (-not $p.WaitForExit(300000)) { $p.Kill(); Write-Host "  (killed, took too long)" }
    }
}
Remove-Item $v1 -ErrorAction SilentlyContinue
Write-Host "Done. See results/runs.csv"
