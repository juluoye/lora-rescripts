$ErrorActionPreference = "Stop"

$Env:HF_HOME = "huggingface"
$Env:PYTHONUTF8 = "1"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $repoRoot "tools\runtime\runtime_paths.ps1")
. (Join-Path $repoRoot "tools\runtime\mirror_env.ps1")

if (Test-MikazukiChinaMirrorMode) {
    Enable-MikazukiChinaMirrorMode -RepoRoot $repoRoot
}

$portableRuntimeInfo = Resolve-RuntimeDirectoryInfo -RepoRoot $repoRoot -RuntimeName "portable"
$portableRuntimeDir = $portableRuntimeInfo.DirectoryPath
$portablePython = Join-Path $portableRuntimeDir "python.exe"
$portableMarker = Join-Path $portableRuntimeDir ".deps_installed"

$venvRuntimeInfo = Resolve-RuntimeDirectoryInfo -RepoRoot $repoRoot -RuntimeName "venv"
$venvRuntimeDir = $venvRuntimeInfo.DirectoryPath
$venvPython = Join-Path $venvRuntimeDir "Scripts\python.exe"
$venvMarker = Join-Path $venvRuntimeDir ".deps_installed"
$allowExternalPython = $Env:MIKAZUKI_ALLOW_SYSTEM_PYTHON -eq "1"
$mainRequiredModules = @("accelerate", "torch", "fastapi", "toml", "transformers", "diffusers", "peft", "torchdiffeq", "timm", "lion_pytorch", "dadaptation", "schedulefree", "prodigyopt", "prodigyplus", "pytorch_optimizer", "tensorboard", "pkg_resources", "triton")

function Test-PipReady {
    param (
        [string]$PythonExe
    )

    $process = New-Object System.Diagnostics.Process
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $PythonExe
    $startInfo.Arguments = "-m pip --version"
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process.StartInfo = $startInfo

    try {
        $null = $process.Start()
        $process.WaitForExit()
        return $process.ExitCode -eq 0
    }
    finally {
        $process.Dispose()
    }
}

function Invoke-Step {
    param (
        [string]$Message,
        [scriptblock]$Action
    )

    Write-Host -ForegroundColor Green $Message
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Message failed with exit code $LASTEXITCODE."
    }
}

function Invoke-OptionalStep {
    param (
        [string]$Message,
        [scriptblock]$Action,
        [string]$WarningMessage
    )

    Write-Host -ForegroundColor Green $Message
    & $Action
    if ($LASTEXITCODE -ne 0) {
        Write-Host -ForegroundColor Yellow $WarningMessage
    }
}

function Test-ModulesReady {
    param (
        [string]$PythonExe,
        [string[]]$Modules
    )

    if (-not $Modules -or $Modules.Count -eq 0) {
        return $true
    }

    & $PythonExe -c "import importlib, sys, warnings; warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API.*', category=UserWarning); failed=[]; 
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception:
        failed.append(name)
raise SystemExit(1 if failed else 0)" @Modules 1>$null 2>$null
    return $LASTEXITCODE -eq 0
}

function Invoke-PythonJsonProbe {
    param (
        [string]$PythonExe,
        [string]$ScriptContent
    )

    $tempPath = [System.IO.Path]::GetTempFileName()
    $tempPyPath = [System.IO.Path]::ChangeExtension($tempPath, ".py")
    Move-Item -LiteralPath $tempPath -Destination $tempPyPath -Force

    try {
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($tempPyPath, $ScriptContent, $utf8NoBom)
        $raw = & $PythonExe $tempPyPath 2>$null
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($raw)) {
            return $null
        }

        $text = if ($raw -is [System.Array]) {
            ($raw | ForEach-Object { [string]$_ }) -join [Environment]::NewLine
        }
        else {
            [string]$raw
        }

        $jsonLine = $text -split "\r?\n" |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -match '^[\{\[]' } |
            Select-Object -Last 1

        if ([string]::IsNullOrWhiteSpace($jsonLine)) {
            return $null
        }

        try {
            return $jsonLine | ConvertFrom-Json
        }
        catch {
            return $null
        }
    }
    finally {
        Remove-Item -LiteralPath $tempPyPath -Force -ErrorAction SilentlyContinue
    }
}

function Get-PythonRuntimeVersionInfo {
    param (
        [string]$PythonExe
    )

    $script = @"
import json
import sys

print(json.dumps({
    "version": sys.version.split()[0],
    "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    "abi_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
    "python_lib_name": f"python{sys.version_info.major}{sys.version_info.minor}.lib",
}))
"@

    return Invoke-PythonJsonProbe -PythonExe $PythonExe -ScriptContent $script
}

function Ensure-EmbeddablePythonDevFiles {
    param (
        [string]$PythonExe,
        [string]$RuntimeDir
    )

    $pthFile = Get-ChildItem -Path $RuntimeDir -Filter 'python*._pth' -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $pthFile) {
        return
    }

    $versionInfo = Get-PythonRuntimeVersionInfo -PythonExe $PythonExe
    if (-not $versionInfo) {
        throw "Could not determine Python runtime version for $RuntimeDir."
    }

    $pythonLibName = if ($versionInfo.python_lib_name) { [string]$versionInfo.python_lib_name } else { "python$($versionInfo.major)$($versionInfo.minor).lib" }
    $runtimeIncludeDir = Join-Path $RuntimeDir "Include"
    $runtimeLibDir = Join-Path $RuntimeDir "libs"
    $runtimePythonHeader = Join-Path $runtimeIncludeDir "Python.h"
    $runtimePythonLib = Join-Path $runtimeLibDir $pythonLibName

    if ((Test-Path $runtimePythonHeader) -and (Test-Path $runtimePythonLib)) {
        Write-Host -ForegroundColor Green "Python dev files already present for runtime."
        return
    }

    $devCacheRoot = Join-Path $repoRoot ".python-dev-cache"
    $nugetVersion = $versionInfo.version
    $packageFile = Join-Path $devCacheRoot "python.$nugetVersion.nupkg"
    $extractDir = Join-Path $devCacheRoot "python-$nugetVersion"
    $packageUrl = "https://www.nuget.org/api/v2/package/python/$nugetVersion"

    if (-not (Test-Path $devCacheRoot)) {
        New-Item -ItemType Directory -Path $devCacheRoot -Force | Out-Null
    }

    if (-not (Test-Path $packageFile)) {
        Write-Host -ForegroundColor Yellow "Downloading CPython dev package for $nugetVersion..."
        Invoke-WebRequest -Uri $packageUrl -OutFile $packageFile
    }

    $sourceIncludeDir = Join-Path $extractDir "tools\include"
    $sourcePythonLib = Join-Path $extractDir "tools\libs\$pythonLibName"

    if (-not (Test-Path $sourceIncludeDir) -or -not (Test-Path $sourcePythonLib)) {
        if (Test-Path $extractDir) {
            Remove-Item -LiteralPath $extractDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        Write-Host -ForegroundColor Yellow "Extracting CPython dev package for $nugetVersion..."
        $zipPackageFile = Join-Path $devCacheRoot "python.$nugetVersion.zip"
        Copy-Item -LiteralPath $packageFile -Destination $zipPackageFile -Force
        Expand-Archive -LiteralPath $zipPackageFile -DestinationPath $extractDir -Force
    }

    if (-not (Test-Path $sourceIncludeDir) -or -not (Test-Path $sourcePythonLib)) {
        throw "Downloaded CPython dev package does not contain the expected Include or $pythonLibName files."
    }

    if (-not (Test-Path $runtimeIncludeDir)) {
        New-Item -ItemType Directory -Path $runtimeIncludeDir -Force | Out-Null
    }
    if (-not (Test-Path $runtimeLibDir)) {
        New-Item -ItemType Directory -Path $runtimeLibDir -Force | Out-Null
    }

    Copy-Item -Path (Join-Path $sourceIncludeDir "*") -Destination $runtimeIncludeDir -Recurse -Force
    Copy-Item -LiteralPath $sourcePythonLib -Destination $runtimePythonLib -Force
    Write-Host -ForegroundColor Green "Provisioned Python dev files into runtime."
}

function Get-TritonRuntimeProbe {
    param (
        [string]$PythonExe
    )

    $script = @"
import json
import sys
import importlib.metadata as md

result = {
    "python_version": sys.version.split()[0],
    "torch_version": "",
    "torchvision_version": "",
    "triton_version": "",
    "cuda_available": False,
    "triton_import_ok": False,
    "triton_error": "",
}

def metadata_version(*names):
    for name in names:
        try:
            return md.version(name)
        except Exception:
            continue
    return ""

try:
    import torch
except Exception as exc:
    result["triton_error"] = f"torch import failed: {exc}"
    print(json.dumps(result))
    raise SystemExit(0)

result["torch_version"] = getattr(torch, "__version__", "")
result["torchvision_version"] = metadata_version("torchvision")
result["triton_version"] = metadata_version("triton-windows", "triton")
result["cuda_available"] = bool(torch.cuda.is_available())

try:
    import triton  # noqa: F401
    result["triton_import_ok"] = True
except Exception as exc:
    result["triton_error"] = str(exc)

print(json.dumps(result))
"@

    return Invoke-PythonJsonProbe -PythonExe $PythonExe -ScriptContent $script
}

function Assert-TritonRuntimeReady {
    param (
        [string]$PythonExe,
        [string]$ExpectedTritonVersion = "3.6.0.post26"
    )

    $probe = Get-TritonRuntimeProbe -PythonExe $PythonExe
    if (-not $probe) {
        throw "Could not probe runtime details after Triton installation."
    }

    $issues = New-Object System.Collections.Generic.List[string]
    if ($ExpectedTritonVersion -and $probe.triton_version -ne $ExpectedTritonVersion) {
        $issues.Add("triton is $($probe.triton_version), expected $ExpectedTritonVersion") | Out-Null
    }
    if (-not $probe.triton_import_ok) {
        $errorMessage = $probe.triton_error
        if ([string]::IsNullOrWhiteSpace($errorMessage)) {
            $errorMessage = "triton import failed"
        }
        $issues.Add($errorMessage) | Out-Null
    }

    if ($issues.Count -gt 0) {
        throw "Triton runtime verification failed: $($issues -join '; ')"
    }

    Write-Host -ForegroundColor Green "Triton runtime versions: Python $($probe.python_version); Torch $($probe.torch_version); TorchVision $($probe.torchvision_version); Triton $($probe.triton_version)"
    Write-Host -ForegroundColor Green "CUDA available: $($probe.cuda_available)"
}

if (Test-Path $portablePython) {
    Write-Host -ForegroundColor Green "Using portable Python..."
    if (-not (Test-PipReady -PythonExe $portablePython)) {
        throw @"
Portable Python is incomplete: pip is not available.

This project now assumes the bundled python folder is already a ready-to-run environment for packaging and distribution.
Normal installation will not auto-bootstrap embeddable Python anymore.

Recommended fix:
1. Replace the bundled python folder with a prepared portable Python environment.
2. If you are repairing a raw embeddable Python manually, run setup_embeddable_python.bat yourself.
"@
    }
    $pythonExe = $portablePython
    $markerPath = $portableMarker
    $runtimeRoot = $portableRuntimeDir
}
elseif (Test-Path $venvPython) {
    Write-Host -ForegroundColor Green "Using existing project virtual environment..."
    if (-not (Test-PipReady -PythonExe $venvPython)) {
        throw "Project virtual environment is incomplete: pip is not available. Repair or recreate .\venv first."
    }
    $pythonExe = $venvPython
    $markerPath = $venvMarker
    $runtimeRoot = $venvRuntimeDir
}
elseif ($allowExternalPython) {
    Write-Host -ForegroundColor Yellow "No project-local Python found. MIKAZUKI_ALLOW_SYSTEM_PYTHON=1 is set, creating a project-local venv from system Python..."
    $venvParentDir = Split-Path -Parent $venvRuntimeDir
    if (-not (Test-Path $venvParentDir)) {
        New-Item -ItemType Directory -Path $venvParentDir -Force | Out-Null
    }
    python -m venv $venvRuntimeDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create venv."
    }
    $pythonExe = $venvPython
    $markerPath = $venvMarker
    $runtimeRoot = $venvRuntimeDir
}
else {
    throw @"
No project-local Python environment was found.

This installer is locked to project-local Python by default to avoid leaking packages into the host machine.

Expected one of:
- $portablePython
- $venvPython

Recommended fix:
1. Bundle a ready-to-run portable Python in .\env\python or the legacy .\python
2. Or set MIKAZUKI_ALLOW_SYSTEM_PYTHON=1 once to bootstrap a project-local .\env\venv or legacy .\venv for development
"@
}

Set-Location $repoRoot

$runtimeCacheRoot = Get-MikazukiRuntimeDependencyCacheDir -RepoRoot $repoRoot -RuntimeId "standard"

Invoke-Step "Provisioning Python dev files required by Triton..." {
    Ensure-EmbeddablePythonDevFiles -PythonExe $pythonExe -RuntimeDir $runtimeRoot
}

Invoke-Step "Upgrading pip tooling..." {
    & $pythonExe -m pip install --upgrade --no-warn-script-location pip "setuptools<81" wheel
}

Invoke-Step "Installing PyTorch and torchvision (CUDA 12.8 channel)..." {
    $mirrorArgs = @(
        "--upgrade",
        "--no-warn-script-location",
        "--prefer-binary",
        "torch==2.10.0+cu128",
        "torchvision==0.25.0+cu128"
    )
    $mirrorArgs = Add-MikazukiRuntimeCacheArgs -PipArgs $mirrorArgs -RepoRoot $repoRoot -RuntimeId "standard" -ItemIds @("torch_stack")
    $fallbackArgs = $mirrorArgs + @("--extra-index-url", "https://download.pytorch.org/whl/cu128")
    Invoke-MirrorAwarePipInstall `
        -PythonExe $pythonExe `
        -MirrorArgs $mirrorArgs `
        -FallbackArgs $fallbackArgs `
        -MirrorLabel "China mirror (PyPI + SJTU PyTorch wheel mirror)" `
        -FallbackLabel "official PyTorch CUDA 12.8 channel" | Out-Null
}

Invoke-OptionalStep "Installing xformers (optional)..." {
    $mirrorArgs = @(
        "--upgrade",
        "--no-warn-script-location",
        "--only-binary",
        "xformers",
        "xformers>=0.0.34"
    )
    $mirrorArgs = Add-MikazukiRuntimeCacheArgs -PipArgs $mirrorArgs -RepoRoot $repoRoot -RuntimeId "standard" -ItemIds @("xformers")
    $fallbackArgs = $mirrorArgs + @("--index-url", "https://download.pytorch.org/whl/cu128")
    Invoke-MirrorAwarePipInstall `
        -PythonExe $pythonExe `
        -MirrorArgs $mirrorArgs `
        -FallbackArgs $fallbackArgs `
        -MirrorLabel "China mirror (PyPI + SJTU PyTorch wheel mirror)" `
        -FallbackLabel "official PyTorch CUDA 12.8 xformers channel" | Out-Null
} "Optional xformers installation failed. The GUI will still work and training can fall back to SDPA."

Invoke-Step "Installing project dependencies..." {
    $requirementArgs = @(
        "--upgrade",
        "--no-warn-script-location",
        "--prefer-binary",
        "-r",
        "requirements.txt"
    )
    $requirementArgs = Add-MikazukiRuntimeCacheArgs -PipArgs $requirementArgs -RepoRoot $repoRoot -RuntimeId "standard" -ItemIds @("requirements")
    & $pythonExe -m pip install @requirementArgs
}

Invoke-Step "Installing Triton runtime..." {
    $tritonArgs = @(
        "--upgrade",
        "--no-warn-script-location",
        "--prefer-binary",
        "triton-windows==3.6.0.post26"
    )
    $tritonArgs = Add-MikazukiRuntimeCacheArgs -PipArgs $tritonArgs -RepoRoot $repoRoot -RuntimeId "standard" -ItemIds @("triton_runtime_default")
    & $pythonExe -m pip install @tritonArgs
}

Invoke-Step "Re-enabling pkg_resources compatibility for TensorBoard..." {
    & $pythonExe -m pip install --upgrade --no-warn-script-location --prefer-binary "setuptools<81" 2>&1
}

if (-not (Test-ModulesReady -PythonExe $pythonExe -Modules $mainRequiredModules)) {
    throw "Project dependencies did not finish installing correctly. One or more required runtime modules are still missing."
}

Invoke-Step "Verifying Triton runtime..." {
    Assert-TritonRuntimeReady -PythonExe $pythonExe
}

if ($markerPath) {
    Set-Content -Path $markerPath -Value "" -Encoding ASCII
}

Write-Host -ForegroundColor Green "Install completed"
