# run_wlf_osc_pipeline.ps1
# Launch two Orbbec feet publishers + OSC blob fuser with clearly delineated variables.
# Stop with Ctrl+C in this window; all child processes will be terminated.

# -----------------------------
# USER CONFIG
# -----------------------------

$PYTHON = "python"
$PROJECT_ROOT = "C:\Users\jeffk\PycharmProjects\wlf2026"

$FEET_SCRIPT  = "src\feet_xy_osc_femto_bolt_validated.py"
$FUSER_SCRIPT = "src\osc_blob_fuser.py"

$CAM0_INDEX = 0
$CAM1_INDEX = 1

$CAM0_PORT = 9001
$CAM1_PORT = 9002

$FUSED_OUT_HOST = "127.0.0.1"
$FUSED_OUT_PORT = 9100
$FUSED_OUT_ADDR = "/wlf/fused"

$IN_ADDR_BLOBS = "/wlf/blobs"
$IN_ADDR_FRAME = "/wlf/frame"

$MAX_BLOBS_PER_CAM = 8
$MAX_TOTAL_BLOBS   = 16

$CAM0_OFFSET_X = 0.0
$CAM0_OFFSET_Y = 0.0
$CAM0_YAW_DEG  = 0.0

$CAM1_OFFSET_X = 2.0
$CAM1_OFFSET_Y = 0.0
$CAM1_YAW_DEG  = 0.0

$BOUND_XMIN = -1.0
$BOUND_XMAX =  3.0
$BOUND_YMIN = -2.0
$BOUND_YMAX =  2.0

$FEET_VISUALIZE = $true
$FEET_VERBOSE   = $false

$LOG_DIR = "$PROJECT_ROOT\logs"

# -----------------------------
# END USER CONFIG
# -----------------------------

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir($p) {
  if (!(Test-Path -LiteralPath $p)) {
    New-Item -ItemType Directory -Path $p | Out-Null
  }
}

Push-Location $PROJECT_ROOT
Ensure-Dir $LOG_DIR

$feetCommon = @(
  "--osc-enabled",
  "--osc-host", "127.0.0.1",
  "--osc-addr-blobs", $IN_ADDR_BLOBS,
  "--osc-addr-frame", $IN_ADDR_FRAME,
  "--max-blobs", "$MAX_BLOBS_PER_CAM"
)
if ($FEET_VISUALIZE) { $feetCommon += "--visualize" }
if ($FEET_VERBOSE)   { $feetCommon += "--verbose" }

$cam0Args = @($FEET_SCRIPT, "--device-index", "$CAM0_INDEX", "--osc-port", "$CAM0_PORT") + $feetCommon
$cam1Args = @($FEET_SCRIPT, "--device-index", "$CAM1_INDEX", "--osc-port", "$CAM1_PORT") + $feetCommon

$fuserArgs = @(
  $FUSER_SCRIPT,
  "--in",     "cam0:${CAM0_PORT}", "cam1:${CAM1_PORT}",
  "--offset","cam0:${CAM0_OFFSET_X}:${CAM0_OFFSET_Y}", "cam1:${CAM1_OFFSET_X}:${CAM1_OFFSET_Y}",
  "--yaw",   "cam0:${CAM0_YAW_DEG}", "cam1:${CAM1_YAW_DEG}",
  "--bounds", "$BOUND_XMIN", "$BOUND_XMAX", "$BOUND_YMIN", "$BOUND_YMAX",
  "--max-blobs-per-cam", "$MAX_BLOBS_PER_CAM",
  "--max-total-blobs", "$MAX_TOTAL_BLOBS",
  "--in-addr-blobs", $IN_ADDR_BLOBS,
  "--out-host", $FUSED_OUT_HOST,
  "--out-port", "$FUSED_OUT_PORT",
  "--out-addr", $FUSED_OUT_ADDR
)

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logCam0  = Join-Path $LOG_DIR "feet_cam0_$ts.out.log"
$errCam0  = Join-Path $LOG_DIR "feet_cam0_$ts.err.log"
$logCam1  = Join-Path $LOG_DIR "feet_cam1_$ts.out.log"
$errCam1  = Join-Path $LOG_DIR "feet_cam1_$ts.err.log"
$logFuser = Join-Path $LOG_DIR "fuser_$ts.out.log"
$errFuser = Join-Path $LOG_DIR "fuser_$ts.err.log"

Write-Host "Launching cam0 -> UDP $CAM0_PORT (device-index=$CAM0_INDEX)"
$procCam0 = Start-Process -FilePath $PYTHON -ArgumentList $cam0Args -NoNewWindow -PassThru `
  -RedirectStandardOutput $logCam0 -RedirectStandardError $errCam0

Start-Sleep -Milliseconds 300

Write-Host "Launching cam1 -> UDP $CAM1_PORT (device-index=$CAM1_INDEX)"
$procCam1 = Start-Process -FilePath $PYTHON -ArgumentList $cam1Args -NoNewWindow -PassThru `
  -RedirectStandardOutput $logCam1 -RedirectStandardError $errCam1

Start-Sleep -Milliseconds 300

Write-Host "Launching fuser -> ${FUSED_OUT_HOST}:${FUSED_OUT_PORT} ${FUSED_OUT_ADDR}"
$procFuser = Start-Process -FilePath $PYTHON -ArgumentList $fuserArgs -NoNewWindow -PassThru `
  -RedirectStandardOutput $logFuser -RedirectStandardError $errFuser

Write-Host ""
Write-Host "Running. Logs:"
Write-Host "  cam0  out: $logCam0"
Write-Host "        err: $errCam0"
Write-Host "  cam1  out: $logCam1"
Write-Host "        err: $errCam1"
Write-Host "  fuser out: $logFuser"
Write-Host "        err: $errFuser"
Write-Host ""
Write-Host "Press Ctrl+C to stop all processes."

$stopping = $false
$handler = {
  if ($script:stopping) { return }
  $script:stopping = $true
  Write-Host "`nStopping..."
  foreach ($p in @($script:procFuser, $script:procCam1, $script:procCam0)) {
    try {
      if ($p -and !$p.HasExited) { Stop-Process -Id $p.Id -Force }
    } catch {}
  }
  Pop-Location
  exit 0
}

$null = Register-EngineEvent -SourceIdentifier ConsoleBreak -Action $handler

while ($true) {
  Start-Sleep -Milliseconds 500
  if ($procCam0.HasExited -or $procCam1.HasExited -or $procFuser.HasExited) {
    Write-Host "`nA process exited unexpectedly."
    & $handler
  }
}
