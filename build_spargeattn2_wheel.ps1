param(
    [string]$PythonExe = "",
    [string]$TorchCudaArchList = "8.6;8.9;9.0+PTX",
    [string]$CudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    [switch]$UseNinja,
    [switch]$CleanFirst,
    [switch]$SkipVsDevShell
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$spargeRoot = Join-Path $repoRoot "ref\SpargeAttn-main"
$wheelOutDir = Join-Path $repoRoot "wheel"
$runtimePython = Join-Path $repoRoot "env\python-spargeattn2\Scripts\python.exe"
$shortBuildRoot = "H:\tmp\sa2w"
$shortBuildBase = Join-Path $shortBuildRoot "b"
$shortDistDir = Join-Path $shortBuildRoot "d"
$shortBuildLib = Join-Path $shortBuildBase "lib.win-amd64"
$vcvarsCandidates = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = $runtimePython
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $spargeRoot)) {
    throw "SpargeAttn source tree not found: $spargeRoot"
}
if (-not (Test-Path (Join-Path $CudaHome "bin\nvcc.exe"))) {
    throw "CUDA nvcc not found under: $CudaHome"
}

function Get-VcVarsPath {
    foreach ($candidate in $vcvarsCandidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return $null
}

$env:CUDA_HOME = $CudaHome
$env:TORCH_CUDA_ARCH_LIST = $TorchCudaArchList
$env:SPARGEATTN_BUILD_ARCHES = $TorchCudaArchList
$env:SPARGEATTN_ENABLE_SM90_NATIVE = "0"
$env:SPARGEATTN_BUILD_BASE = $shortBuildBase
$env:PYTHONUTF8 = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:DISTUTILS_USE_SDK = "1"

if ($UseNinja) {
    $ninjaPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
    if (Test-Path $ninjaPath) {
        $env:Path = "$ninjaPath;$env:Path"
    }
}

if (-not (Test-Path $wheelOutDir)) {
    New-Item -ItemType Directory -Path $wheelOutDir -Force | Out-Null
}

if ($CleanFirst) {
    foreach ($path in @(
        (Join-Path $spargeRoot "build"),
        (Join-Path $spargeRoot "dist"),
        $shortBuildRoot
    )) {
        if (Test-Path $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
}

foreach ($path in @($shortBuildRoot, $shortBuildBase, $shortDistDir)) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

Write-Host -ForegroundColor Cyan "Building SpargeAttn wheel"
Write-Host -ForegroundColor DarkGray "Python: $PythonExe"
Write-Host -ForegroundColor DarkGray "CUDA_HOME: $CudaHome"
Write-Host -ForegroundColor DarkGray "TORCH_CUDA_ARCH_LIST: $TorchCudaArchList"
Write-Host -ForegroundColor DarkGray "SPARGEATTN_ENABLE_SM90_NATIVE: $env:SPARGEATTN_ENABLE_SM90_NATIVE"
Write-Host -ForegroundColor DarkGray "Short build base: $shortBuildBase"
Write-Host -ForegroundColor DarkGray "Short dist dir: $shortDistDir"

$vcvarsPath = $null
if (-not $SkipVsDevShell) {
    $vcvarsPath = Get-VcVarsPath
    if (-not $vcvarsPath) {
        throw "Could not find vcvars64.bat for Visual Studio Build Tools."
    }
    Write-Host -ForegroundColor DarkGray "MSVC env: $vcvarsPath"
}

Push-Location $spargeRoot
try {
    & $PythonExe -m pip install --upgrade --no-warn-script-location pip "setuptools<81" wheel ninja
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare build tooling."
    }

    & $PythonExe -c "import torch" 1>$null 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host -ForegroundColor Yellow "PyTorch is missing in the SpargeAttn build environment. Installing the CUDA 12.8 torch stack first..."
        & $PythonExe -m pip install --upgrade --no-warn-script-location --prefer-binary `
            "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" `
            --extra-index-url "https://download.pytorch.org/whl/cu128"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install the CUDA 12.8 PyTorch build stack."
        }
    }

    & $PythonExe -m pip install --upgrade --no-warn-script-location --prefer-binary `
        packaging einops numpy
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install the minimal Python build/runtime helpers."
    }

    $builtQattn = Join-Path $shortBuildLib "spas_sage_attn\_qattn.cp311-win_amd64.pyd"
    $builtFused = Join-Path $shortBuildLib "spas_sage_attn\_fused.cp311-win_amd64.pyd"
    $canReuseBuild = (Test-Path $builtQattn) -and (Test-Path $builtFused)
    if ($canReuseBuild) {
        Write-Host -ForegroundColor Yellow "Reusing existing short-path build outputs for wheel packaging."
    }

    if ($SkipVsDevShell) {
        if ($canReuseBuild) {
            & $PythonExe setup.py bdist_wheel --skip-build --dist-dir $shortDistDir
        }
        else {
            & $PythonExe setup.py build --build-base $shortBuildBase bdist_wheel --skip-build --dist-dir $shortDistDir
        }
        if ($LASTEXITCODE -ne 0) {
            throw "SpargeAttn wheel build failed."
        }
    }
    else {
        $pythonExeForCmd = $PythonExe.Replace('"', '\"')
        $spargeRootForCmd = $spargeRoot.Replace('"', '\"')
        $shortDistDirForCmd = $shortDistDir.Replace('"', '\"')
        if ($canReuseBuild) {
            $cmd = "`"$vcvarsPath`" && cd /d `"$spargeRootForCmd`" && `"$pythonExeForCmd`" setup.py bdist_wheel --skip-build --dist-dir `"$shortDistDirForCmd`""
        }
        else {
            $shortBuildBaseForCmd = $shortBuildBase.Replace('"', '\"')
            $cmd = "`"$vcvarsPath`" && cd /d `"$spargeRootForCmd`" && `"$pythonExeForCmd`" setup.py build --build-base `"$shortBuildBaseForCmd`" bdist_wheel --skip-build --dist-dir `"$shortDistDirForCmd`""
        }
        cmd.exe /c $cmd
        if ($LASTEXITCODE -ne 0) {
            throw "SpargeAttn wheel build failed."
        }
    }

    $builtWheels = Get-ChildItem -LiteralPath $shortDistDir -Filter *.whl -File -ErrorAction SilentlyContinue
    if (-not $builtWheels) {
        throw "No wheel was produced under $shortDistDir"
    }

    foreach ($builtWheel in $builtWheels) {
        Copy-Item -LiteralPath $builtWheel.FullName -Destination (Join-Path $wheelOutDir $builtWheel.Name) -Force
        Write-Host -ForegroundColor Green "Copied wheel: $(Join-Path $wheelOutDir $builtWheel.Name)"
    }
}
finally {
    Pop-Location
}
