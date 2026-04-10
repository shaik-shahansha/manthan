param(
    # Data stages
    [switch]$SkipBuildLocalDataset,   # Skip programmatic 100-sample builder
    [switch]$SkipFormatDataset,       # Skip dataset formatter (synthetic_traces → ChatML)
    [switch]$SkipRewardDataset,       # Skip GRPO reward dataset builder

    # Training stages
    [switch]$SkipSft,                 # Skip SFT training
    [switch]$SkipGrpo,                # Skip GRPO training

    # Eval stages
    [switch]$SkipGsm8kEval,
    [switch]$SkipMbppEval,

    # Config selection
    [string]$SftConfig  = "configs/sft_local_gpu.yaml",   # Full local GPU SFT
    [string]$GrpoConfig = "configs/grpo_local_gpu.yaml",  # Full local GPU GRPO
    [string]$EvalModel  = "outputs/grpo_local_gpu",       # Model to evaluate

    # Misc
    [int]$EvalSamples = 10
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe"
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Host "`n==> $Name" -ForegroundColor Cyan
    & $pythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $repoRoot
try {
    # ── Data ──────────────────────────────────────────────────────────────────

    if (-not $SkipBuildLocalDataset) {
        Invoke-Step -Name "Build local programmatic dataset (100 diverse traces)" -Arguments @(
            "src/data/build_local_dataset.py",
            "--output", "data/raw/local_traces.jsonl"
        )
    }

    if (-not $SkipFormatDataset) {
        # Merge local_traces.jsonl with any existing synthetic_traces.jsonl
        # Format the combined dataset into ChatML HuggingFace Dataset
        $inputFile = "data/raw/local_traces.jsonl"
        if (Test-Path "data/raw/synthetic_traces.jsonl") {
            # Merge both files into a temp combined file
            $combined = "data/raw/combined_traces.jsonl"
            Get-Content "data/raw/local_traces.jsonl", "data/raw/synthetic_traces.jsonl" |
                Set-Content $combined
            $inputFile = $combined
        }
        Invoke-Step -Name "Format dataset (local_traces → ChatML HuggingFace Dataset)" -Arguments @(
            "src/data/format_dataset.py",
            "--input", $inputFile,
            "--output", "data/processed/manthan_dataset"
        )
    }

    if (-not $SkipRewardDataset) {
        Invoke-Step -Name "Build GRPO reward dataset (GSM8K + MBPP + TriviaQA)" -Arguments @(
            "src/data/reward_dataset.py",
            "--output", "data/processed/reward_dataset.jsonl"
        )
    }

    # ── Training ──────────────────────────────────────────────────────────────

    if (-not $SkipSft) {
        Write-Host "`n==> SFT training — config: $SftConfig" -ForegroundColor Cyan
        & $pythonExe "src/training/sft_train.py" "--config" $SftConfig
        if ($LASTEXITCODE -ne 0) { throw "SFT training failed" }
    }

    if (-not $SkipGrpo) {
        # Default GRPO model comes from SFT output dir in the grpo config
        Write-Host "`n==> GRPO training — config: $GrpoConfig" -ForegroundColor Cyan
        & $pythonExe "src/training/grpo_train.py" "--config" $GrpoConfig
        if ($LASTEXITCODE -ne 0) { throw "GRPO training failed" }
    }

    # ── Eval ──────────────────────────────────────────────────────────────────

    if (-not $SkipGsm8kEval) {
        Invoke-Step -Name "GSM8K evaluation ($EvalSamples samples)" -Arguments @(
            "src/eval/benchmark_gsm8k.py",
            "--model", $EvalModel,
            "--n-samples", $EvalSamples.ToString(),
            "--output", "outputs/eval/gsm8k_local.json"
        )
    }

    if (-not $SkipMbppEval) {
        Invoke-Step -Name "MBPP evaluation ($EvalSamples samples)" -Arguments @(
            "src/eval/benchmark_mbpp.py",
            "--model", $EvalModel,
            "--n-samples", $EvalSamples.ToString(),
            "--output", "outputs/eval/mbpp_local.json"
        )
    }

    Write-Host "`nLocal pipeline completed successfully." -ForegroundColor Green
    Write-Host "  SFT output:   outputs/sft_local_gpu/" -ForegroundColor Gray
    Write-Host "  GRPO output:  outputs/grpo_local_gpu/" -ForegroundColor Gray
    Write-Host "  Eval results: outputs/eval/" -ForegroundColor Gray
}
finally {
    Pop-Location
}

