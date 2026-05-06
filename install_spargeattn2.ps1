param(
    [string]$SpargeAttnWheel = "",
    [string]$SpargeAttnVersion = "0.1.0"
)

$ErrorActionPreference = "Stop"

$Env:HF_HOME = "huggingface"
$Env:PYTHONUTF8 = "1"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$null = . (Join-Path $repoRoot "tools\runtime\runtime_paths.ps1")
. (Join-Path $repoRoot "tools\runtime\mirror_env.ps1")

if (Test-MikazukiChinaMirrorMode) {
    Enable-MikazukiChinaMirrorMode -RepoRoot $repoRoot
}

$runtimeInfo = Resolve-RuntimeDirectoryInfo -RepoRoot $repoRoot -RuntimeName "spargeattn2"
$runtimeDirName = $runtimeInfo.DirectoryName
$runtimeDir = $runtimeInfo.DirectoryPath
$runtimePython = Join-Path $runtimeDir "python.exe"
if (-not (Test-Path $runtimePython)) {
    $runtimePython = Join-Path $runtimeDir "Scripts\python.exe"
}
$runtimeMarker = Join-Path $runtimeDir ".deps_installed"
$mainRequiredModules = @("accelerate", "torch", "fastapi", "toml", "transformers", "diffusers", "peft", "torchdiffeq", "timm", "lion_pytorch", "dadaptation", "schedulefree", "prodigyopt", "prodigyplus", "pytorch_optimizer", "tensorboard", "safetensors", "huggingface_hub", "imagesize", "cv2", "einops", "google.protobuf", "sentencepiece", "numpy", "gradio", "imageio", "imageio_ffmpeg", "triton", "requests", "PIL", "packaging", "psutil", "lycoris", "rich")
$requirementsPath = Join-Path $repoRoot "requirements_spargeattn2_extra.txt"

function Test-PipReady {
    param (
        [string]$PythonExe
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonExe -m pip --version 1>$null 2>$null
        return $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
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

function Test-ModulesReady {
    param (
        [string]$PythonExe,
        [string[]]$Modules
    )

    if (-not $Modules -or $Modules.Count -eq 0) {
        return $true
    }

    & $PythonExe -c "import sys, importlib; failed=[];`nfor name in sys.argv[1:]:`n    try:`n        importlib.import_module(name)`n    except Exception:`n        failed.append(name)`nraise SystemExit(1 if failed else 0)" @Modules 1>$null 2>$null
    return $LASTEXITCODE -eq 0
}

function Get-PythonRuntimeVersionInfo {
    param (
        [string]$PythonExe
    )

    $raw = & $PythonExe -c "import json,sys; print(json.dumps({'major':sys.version_info.major,'minor':sys.version_info.minor,'micro':sys.version_info.micro,'abi_tag':f'cp{sys.version_info.major}{sys.version_info.minor}','version':f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'}))"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($raw)) {
        return $null
    }
    return ($raw | ConvertFrom-Json)
}

function Ensure-EmbeddablePythonDevFiles {
    param (
        [string]$PythonExe,
        [string]$RuntimeDir
    )

    $versionInfo = Get-PythonRuntimeVersionInfo -PythonExe $PythonExe
    if (-not $versionInfo) {
        throw "Could not determine Python runtime version for $RuntimeDir."
    }

    $pythonLibName = "python$($versionInfo.major)$($versionInfo.minor).lib"
    $runtimeIncludeDir = Join-Path $RuntimeDir "Include"
    $runtimeLibDir = Join-Path $RuntimeDir "libs"
    $runtimePythonHeader = Join-Path $runtimeIncludeDir "Python.h"
    $runtimePythonLib = Join-Path $runtimeLibDir $pythonLibName

    if ((Test-Path $runtimePythonHeader) -and (Test-Path $runtimePythonLib)) {
        Write-Host -ForegroundColor Green "Python dev files already present for $runtimeDirName."
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
    Write-Host -ForegroundColor Green "Provisioned Python dev files into $runtimeDirName."
}

function Get-GitHubMirrorCandidates {
    $candidates = @()

    $customMirror = ([string]$Env:MIKAZUKI_GITHUB_MIRROR_BASE).Trim()
    if (-not [string]::IsNullOrWhiteSpace($customMirror)) {
        if (-not $customMirror.EndsWith('/')) {
            $customMirror += '/'
        }
        $candidates += $customMirror
    }

    $defaultMirror = 'https://hub.gitmirror.com/https://github.com/'
    if (-not ($candidates -contains $defaultMirror)) {
        $candidates += $defaultMirror
    }

    return @($candidates)
}

function Join-GitHubMirrorUrl {
    param(
        [string]$MirrorBase,
        [string]$GitHubUrl
    )

    $normalizedMirrorBase = ([string]$MirrorBase).Trim()
    $normalizedGitHubUrl = ([string]$GitHubUrl).Trim()
    if ([string]::IsNullOrWhiteSpace($normalizedMirrorBase) -or [string]::IsNullOrWhiteSpace($normalizedGitHubUrl)) {
        return $normalizedGitHubUrl
    }

    if (-not $normalizedMirrorBase.EndsWith('/')) {
        $normalizedMirrorBase += '/'
    }

    $githubPrefix = 'https://github.com/'
    if (
        $normalizedGitHubUrl.StartsWith($githubPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and
        $normalizedMirrorBase.EndsWith($githubPrefix, [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        return ($normalizedMirrorBase + $normalizedGitHubUrl.Substring($githubPrefix.Length))
    }

    if ($normalizedGitHubUrl -match '^https?://') {
        return ($normalizedMirrorBase + $normalizedGitHubUrl)
    }

    return ($normalizedMirrorBase + $normalizedGitHubUrl.TrimStart('/'))
}

function Get-SpargeAttnWheelUrlCandidates {
    param(
        [string]$WheelUrl
    )

    if ([string]::IsNullOrWhiteSpace($WheelUrl)) {
        return @()
    }

    if (-not ($WheelUrl -match '^https?://')) {
        return @($WheelUrl)
    }

    $candidates = @()
    $isGitHubReleaseUrl = $WheelUrl.StartsWith('https://github.com/', [System.StringComparison]::OrdinalIgnoreCase)

    if ($isGitHubReleaseUrl) {
        $mirrorBases = Get-GitHubMirrorCandidates
        if (Test-MikazukiChinaMirrorMode) {
            foreach ($mirrorBase in $mirrorBases) {
                $candidates += (Join-GitHubMirrorUrl -MirrorBase $mirrorBase -GitHubUrl $WheelUrl)
            }
            $candidates += $WheelUrl
        }
        else {
            $candidates += $WheelUrl
            foreach ($mirrorBase in $mirrorBases) {
                $candidates += (Join-GitHubMirrorUrl -MirrorBase $mirrorBase -GitHubUrl $WheelUrl)
            }
        }
    }
    else {
        $candidates += $WheelUrl
    }

    return @($candidates | Select-Object -Unique)
}

function Get-DefaultSpargeAttnWheelUrl {
    param(
        [string]$PythonExe
    )

    $versionInfo = Get-PythonRuntimeVersionInfo -PythonExe $PythonExe
    if (-not $versionInfo) {
        throw "Could not determine Python ABI tag for $runtimeDirName."
    }

    $fileName = "spas_sage_attn-$SpargeAttnVersion-$($versionInfo.abi_tag)-$($versionInfo.abi_tag)-win_amd64.whl"
    $encodedFileName = [System.Uri]::EscapeDataString($fileName)
    return "https://github.com/WhitecrowAurora/lora-rescripts/releases/download/spargeattn2-wheels/$encodedFileName"
}

function Resolve-SpargeAttnWheel {
    param (
        [string]$RequestedWheel,
        [string]$PythonExe
    )

    if ($RequestedWheel) {
        if (Test-Path $RequestedWheel) {
            return (Resolve-Path $RequestedWheel).Path
        }
        if (-not ($RequestedWheel -match '^https?://')) {
            throw "Specified SpargeAttnWheel path was not found: $RequestedWheel"
        }
        return $RequestedWheel
    }

    $versionInfo = Get-PythonRuntimeVersionInfo -PythonExe $PythonExe
    if (-not $versionInfo) {
        throw "Could not determine Python ABI tag for $runtimeDirName."
    }

    $fileName = "spas_sage_attn-$SpargeAttnVersion-$($versionInfo.abi_tag)-$($versionInfo.abi_tag)-win_amd64.whl"
    $searchRoots = @(
        $repoRoot,
        (Join-Path $repoRoot "wheel"),
        (Join-Path $repoRoot "spargeattn-wheels"),
        (Join-Path $repoRoot "spargeattn_wheels"),
        (Join-Path (Get-MikazukiRuntimeDependencyCacheDir -RepoRoot $repoRoot -RuntimeId "spargeattn2") "spargeattn2_wheel")
    )

    foreach ($root in $searchRoots) {
        if (-not (Test-Path $root)) {
            continue
        }
        $candidate = Join-Path $root $fileName
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return (Get-DefaultSpargeAttnWheelUrl -PythonExe $PythonExe)
}

function Assert-SpargeAttnPythonVersion {
    param(
        [string]$PythonExe
    )

    $versionInfo = Get-PythonRuntimeVersionInfo -PythonExe $PythonExe
    if (-not $versionInfo) {
        throw "Could not determine Python version for $runtimeDirName."
    }

    if (($versionInfo.major -ne 3) -or ($versionInfo.minor -ne 11)) {
        throw @"
SpargeAttn2 requires Python 3.11, but the detected runtime is Python $($versionInfo.version).

Current runtime:
- $runtimeDir

Recommended fix:
1. Remove the current $runtimeDirName runtime directory
2. Extract a Python 3.11 embeddable package into:
   - $runtimeDir
3. Run install_spargeattn2.ps1 again
"@
    }
}

function Download-SpargeAttnWheel {
    param(
        [string]$WheelUrl
    )

    if ([string]::IsNullOrWhiteSpace($WheelUrl) -or -not ($WheelUrl -match '^https?://')) {
        throw "Download-SpargeAttnWheel expects an HTTP(S) URL."
    }

    $downloadDir = Join-Path $repoRoot "wheel"
    if (-not (Test-Path $downloadDir)) {
        New-Item -ItemType Directory -Path $downloadDir | Out-Null
    }

    $fileName = [System.IO.Path]::GetFileName(([System.Uri]$WheelUrl).AbsolutePath)
    $fileName = [System.Uri]::UnescapeDataString($fileName)
    $downloadPath = Join-Path $downloadDir $fileName

    foreach ($candidateUrl in (Get-SpargeAttnWheelUrlCandidates -WheelUrl $WheelUrl)) {
        Write-Host -ForegroundColor Yellow "Trying SpargeAttn wheel download: $candidateUrl"
        try {
            Invoke-WebRequest -Uri $candidateUrl -OutFile $downloadPath -TimeoutSec 120 -UseBasicParsing
            if ((Test-Path $downloadPath) -and ((Get-Item -LiteralPath $downloadPath).Length -gt 0)) {
                return $downloadPath
            }
        }
        catch {
            Write-Host -ForegroundColor Yellow ("SpargeAttn wheel download attempt failed: {0}" -f $_.Exception.Message)
        }
        Remove-Item -LiteralPath $downloadPath -Force -ErrorAction SilentlyContinue
    }

    throw @"
Failed to download the SpargeAttn wheel from all candidate URLs.

You can also place the wheel in one of these locations:
- $repoRoot
- $(Join-Path $repoRoot "wheel")
- $(Join-Path $repoRoot "spargeattn-wheels")
- $(Join-Path $repoRoot "spargeattn_wheels")
"@
}

if (-not (Test-Path $runtimePython)) {
    throw @"
SpargeAttn2 portable Python was not found.

Expected:
- $runtimePython

Recommended fix:
1. Extract a Python 3.11 embeddable package into:
   - $runtimeDir
2. Run install_spargeattn2.ps1 again
"@
}

if (-not (Test-PipReady -PythonExe $runtimePython)) {
    Write-Host -ForegroundColor Yellow "$runtimeDirName is not initialized yet. Running setup_embeddable_python.bat..."
    & (Join-Path $repoRoot "setup_embeddable_python.bat") --auto $runtimeDirName
    if ($LASTEXITCODE -ne 0 -or -not (Test-PipReady -PythonExe $runtimePython)) {
        throw "Failed to initialize $runtimeDirName."
    }
}

Invoke-Step "Checking Python version for SpargeAttn2..." {
    Assert-SpargeAttnPythonVersion -PythonExe $runtimePython
}

Invoke-Step "Provisioning Python dev files required by Triton..." {
    Ensure-EmbeddablePythonDevFiles -PythonExe $runtimePython -RuntimeDir $runtimeDir
}

$resolvedWheel = Resolve-SpargeAttnWheel -RequestedWheel $SpargeAttnWheel -PythonExe $runtimePython

Invoke-Step "Upgrading pip tooling for SpargeAttn2 environment..." {
    & $runtimePython -m pip install --upgrade --no-warn-script-location pip "setuptools<81" wheel ninja
}

if (-not (Test-ModulesReady -PythonExe $runtimePython -Modules $mainRequiredModules)) {
    Invoke-Step "Installing SpargeAttn2 runtime dependencies..." {
        $requirementArgs = @(
            "--upgrade",
            "--no-warn-script-location",
            "--prefer-binary",
            "-r",
            $requirementsPath
        )
        $requirementArgs = Add-MikazukiRuntimeCacheArgs -PipArgs $requirementArgs -RepoRoot $repoRoot -RuntimeId "spargeattn2" -ItemIds @("requirements")
        & $runtimePython -m pip install @requirementArgs
    }
}

Invoke-Step "Installing Triton runtime for $runtimeDirName..." {
    $tritonArgs = @(
        "--upgrade",
        "--no-warn-script-location",
        "--prefer-binary",
        "triton-windows==3.6.0.post26"
    )
    $tritonArgs = Add-MikazukiRuntimeCacheArgs -PipArgs $tritonArgs -RepoRoot $repoRoot -RuntimeId "spargeattn2" -ItemIds @("triton_runtime_default")
    & $runtimePython -m pip install @tritonArgs
}

Invoke-Step "Installing CUDA 12.8 PyTorch stack for SpargeAttn2..." {
    $mirrorArgs = @(
        "--upgrade",
        "--force-reinstall",
        "--no-warn-script-location",
        "--prefer-binary",
        "torch==2.10.0+cu128",
        "torchvision==0.25.0+cu128"
    )
    $mirrorArgs = Add-MikazukiRuntimeCacheArgs -PipArgs $mirrorArgs -RepoRoot $repoRoot -RuntimeId "spargeattn2" -ItemIds @("torch_stack")
    $fallbackArgs = $mirrorArgs + @("--extra-index-url", "https://download.pytorch.org/whl/cu128")
    Invoke-MirrorAwarePipInstall `
        -PythonExe $runtimePython `
        -MirrorArgs $mirrorArgs `
        -FallbackArgs $fallbackArgs `
        -MirrorLabel "China mirror (PyPI + SJTU PyTorch wheel mirror)" `
        -FallbackLabel "official PyTorch CUDA 12.8 channel" | Out-Null
}

if ($resolvedWheel -match '^https?://') {
    Write-Host -ForegroundColor Yellow "Using SpargeAttn wheel URL: $resolvedWheel"
    $resolvedWheel = Download-SpargeAttnWheel -WheelUrl $resolvedWheel
}

Write-Host -ForegroundColor Yellow "Using SpargeAttn wheel: $resolvedWheel"

Invoke-Step "Removing any existing SpargeAttn package..." {
    & $runtimePython -m pip uninstall -y spas_sage_attn
}

Invoke-Step "Installing SpargeAttn wheel from local file..." {
    & $runtimePython -m pip install --upgrade --no-warn-script-location --no-deps $resolvedWheel
}

Invoke-Step "Verifying SpargeAttn import..." {
    & $runtimePython -c "import spas_sage_attn, torch; print('spas_sage_attn ok'); print(torch.__version__)"
}

Set-Content -Path $runtimeMarker -Value "" -Encoding ASCII
Write-Host -ForegroundColor Green "SpargeAttn2 environment is ready"
