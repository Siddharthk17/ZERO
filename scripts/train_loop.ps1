param(
    [string] $Device = "cuda",
    [double] $Days = 31,
    [int] $GamesPerBatch = 128,
    [int] $Simulations = 400,
    [int] $EvalBatchSize = 256,
    [int] $TrainingBatchSize = 1024,
    [int] $ReplayCapacity = 4000000,
    [int] $Channels = 256,
    [int] $Blocks = 12,
    [int] $PolicyChannels = 64,
    [double] $TargetReplayRatio = 4,
    [int] $CandidateInterval = 5000,
    [int] $GateGames = 40,
    [int] $GateSimulations = 64,
    [string] $GateDevice = "cpu",
    [int] $ReplaySaveInterval = 5000,
    [double] $ShutdownTimeout = 300,
    [double] $SelfPlayTimeout = 600,
    [string] $RunStatePath = "checkpoints/zero_x/run_state.json",
    [int] $Seed = 1592614637
)

$ErrorActionPreference = "Continue"

$checkpoint = "checkpoints/zero_x/accepted.pt"

while ($true) {
    $arguments = @()
    if (Test-Path $checkpoint) {
        Write-Host "Resuming ZERO-X from $checkpoint"
        $arguments += @("--resume", $checkpoint)
    } elseif ((Test-Path $RunStatePath) -or (Test-Path "checkpoints/zero_x/replay.pkl")) {
        Write-Host "Resuming ZERO-X from recoverable state"
    } else {
        Write-Host "Starting a fresh ZERO-X run"
        $arguments += "--fresh"
    }
    $arguments += @(
        "--device", $Device,
        "--days", $Days,
        "--games-per-batch", $GamesPerBatch,
        "--simulations", $Simulations,
        "--eval-batch-size", $EvalBatchSize,
        "--training-batch-size", $TrainingBatchSize,
        "--replay-capacity", $ReplayCapacity,
        "--channels", $Channels,
        "--blocks", $Blocks,
        "--policy-channels", $PolicyChannels,
        "--target-replay-ratio", $TargetReplayRatio,
        "--candidate-interval", $CandidateInterval,
        "--gate-games", $GateGames,
        "--gate-simulations", $GateSimulations,
        "--gate-device", $GateDevice,
        "--replay-save-interval", $ReplaySaveInterval,
        "--shutdown-timeout", $ShutdownTimeout,
        "--self-play-timeout", $SelfPlayTimeout,
        "--run-state-path", $RunStatePath,
        "--seed", $Seed
    )

    & python train_master.py @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0 -or $exitCode -eq 130) {
        Write-Host "ZERO-X exited cleanly with code $exitCode"
        break
    }
    Write-Warning "ZERO-X exited with code $exitCode. Restarting in 10 seconds."
    Start-Sleep -Seconds 10
}
