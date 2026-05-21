param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeDir,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

function Invoke-PythonJsonProbe {
    param(
        [string]$PythonExePath,
        [string]$ScriptContent
    )

    $tempPath = [System.IO.Path]::GetTempFileName()
    $tempPyPath = [System.IO.Path]::ChangeExtension($tempPath, ".py")
    Move-Item -LiteralPath $tempPath -Destination $tempPyPath -Force

    try {
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($tempPyPath, $ScriptContent, $utf8NoBom)
        $raw = & $PythonExePath $tempPyPath 2>$null
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
    param(
        [string]$PythonExePath
    )

    $script = @"
import json
import sys

print(json.dumps({
    "major": sys.version_info.major,
    "minor": sys.version_info.minor,
    "micro": sys.version_info.micro,
    "version": sys.version.split()[0],
    "python_lib_name": f"python{sys.version_info.major}{sys.version_info.minor}.lib",
}))
"@

    return Invoke-PythonJsonProbe -PythonExePath $PythonExePath -ScriptContent $script
}

function Resolve-RuntimePythonExe {
    param(
        [string]$RuntimeRoot,
        [string]$RequestedPythonExe
    )

    if (-not [string]::IsNullOrWhiteSpace($RequestedPythonExe) -and (Test-Path $RequestedPythonExe)) {
        return [System.IO.Path]::GetFullPath($RequestedPythonExe)
    }

    foreach ($candidate in @(
        (Join-Path $RuntimeRoot "python.exe"),
        (Join-Path $RuntimeRoot "Scripts\python.exe")
    )) {
        if (Test-Path $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    throw "Python executable not found for runtime: $RuntimeRoot"
}

function Find-LocalDevDonor {
    param(
        [string]$RepoRootPath,
        [string]$TargetRuntimeDir,
        [string]$PythonLibName
    )

    $targetFullPath = [System.IO.Path]::GetFullPath((Join-Path $TargetRuntimeDir "."))
    $candidates = New-Object System.Collections.Generic.List[string]

    $envRoot = Join-Path $RepoRootPath "env"
    if (Test-Path $envRoot) {
        foreach ($dir in Get-ChildItem -LiteralPath $envRoot -Directory -ErrorAction SilentlyContinue) {
            if ($dir.Name -like "python*") {
                $null = $candidates.Add($dir.FullName)
            }
        }
    }

    foreach ($dir in Get-ChildItem -LiteralPath $RepoRootPath -Directory -ErrorAction SilentlyContinue) {
        if ($dir.Name -like "python*") {
            $null = $candidates.Add($dir.FullName)
        }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        $candidateFullPath = [System.IO.Path]::GetFullPath((Join-Path $candidate "."))
        if ($candidateFullPath -eq $targetFullPath) {
            continue
        }

        $candidateHeader = Join-Path $candidateFullPath "Include\Python.h"
        $candidateLibLower = Join-Path $candidateFullPath ("libs\" + $PythonLibName)
        $candidateLibUpper = Join-Path $candidateFullPath ("Libs\" + $PythonLibName)
        if ((Test-Path $candidateHeader) -and ((Test-Path $candidateLibLower) -or (Test-Path $candidateLibUpper))) {
            return $candidateFullPath
        }
    }

    return $null
}

function Copy-DevFilesFromDonor {
    param(
        [string]$DonorRuntimeDir,
        [string]$TargetRuntimeDir,
        [string]$PythonLibName
    )

    $sourceIncludeDir = Join-Path $DonorRuntimeDir "Include"
    $sourcePythonLib = Join-Path $DonorRuntimeDir ("libs\" + $PythonLibName)
    if (-not (Test-Path $sourcePythonLib)) {
        $sourcePythonLib = Join-Path $DonorRuntimeDir ("Libs\" + $PythonLibName)
    }

    if (-not (Test-Path $sourceIncludeDir) -or -not (Test-Path $sourcePythonLib)) {
        throw "Local donor runtime is missing required Python dev files: $DonorRuntimeDir"
    }

    $targetIncludeDir = Join-Path $TargetRuntimeDir "Include"
    $targetLibDir = Join-Path $TargetRuntimeDir "libs"

    if (-not (Test-Path $targetIncludeDir)) {
        New-Item -ItemType Directory -Path $targetIncludeDir -Force | Out-Null
    }
    if (-not (Test-Path $targetLibDir)) {
        New-Item -ItemType Directory -Path $targetLibDir -Force | Out-Null
    }

    Copy-Item -Path (Join-Path $sourceIncludeDir "*") -Destination $targetIncludeDir -Recurse -Force
    Copy-Item -LiteralPath $sourcePythonLib -Destination (Join-Path $targetLibDir $PythonLibName) -Force
}

function Copy-DevFilesFromNuGet {
    param(
        [string]$RepoRootPath,
        [string]$TargetRuntimeDir,
        [string]$Version,
        [string]$PythonLibName
    )

    $devCacheRoot = Join-Path $RepoRootPath ".python-dev-cache"
    $packageFile = Join-Path $devCacheRoot ("python." + $Version + ".nupkg")
    $extractDir = Join-Path $devCacheRoot ("python-" + $Version)
    $zipPackageFile = Join-Path $devCacheRoot ("python." + $Version + ".zip")
    $packageUrl = "https://www.nuget.org/api/v2/package/python/$Version"

    if (-not (Test-Path $devCacheRoot)) {
        New-Item -ItemType Directory -Path $devCacheRoot -Force | Out-Null
    }

    if (-not (Test-Path $packageFile)) {
        Write-Host ("Downloading CPython dev package for " + $Version + "...")
        Invoke-WebRequest -Uri $packageUrl -OutFile $packageFile
    }

    $sourceIncludeDir = Join-Path $extractDir "tools\include"
    $sourcePythonLib = Join-Path $extractDir ("tools\libs\" + $PythonLibName)

    if (-not (Test-Path $sourceIncludeDir) -or -not (Test-Path $sourcePythonLib)) {
        if (Test-Path $extractDir) {
            Remove-Item -LiteralPath $extractDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        Copy-Item -LiteralPath $packageFile -Destination $zipPackageFile -Force
        Expand-Archive -LiteralPath $zipPackageFile -DestinationPath $extractDir -Force
    }

    if (-not (Test-Path $sourceIncludeDir) -or -not (Test-Path $sourcePythonLib)) {
        throw "Downloaded CPython dev package does not contain Include or $PythonLibName."
    }

    $targetIncludeDir = Join-Path $TargetRuntimeDir "Include"
    $targetLibDir = Join-Path $TargetRuntimeDir "libs"
    if (-not (Test-Path $targetIncludeDir)) {
        New-Item -ItemType Directory -Path $targetIncludeDir -Force | Out-Null
    }
    if (-not (Test-Path $targetLibDir)) {
        New-Item -ItemType Directory -Path $targetLibDir -Force | Out-Null
    }

    Copy-Item -Path (Join-Path $sourceIncludeDir "*") -Destination $targetIncludeDir -Recurse -Force
    Copy-Item -LiteralPath $sourcePythonLib -Destination (Join-Path $targetLibDir $PythonLibName) -Force
}

$resolvedRepoRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "."))
$resolvedRuntimeDir = [System.IO.Path]::GetFullPath((Join-Path $RuntimeDir "."))
$resolvedPythonExe = Resolve-RuntimePythonExe -RuntimeRoot $resolvedRuntimeDir -RequestedPythonExe $PythonExe

$versionInfo = Get-PythonRuntimeVersionInfo -PythonExePath $resolvedPythonExe
if (-not $versionInfo) {
    throw "Could not determine Python runtime version for $resolvedRuntimeDir"
}

$pythonLibName = [string]$versionInfo.python_lib_name
$runtimeHeader = Join-Path $resolvedRuntimeDir "Include\Python.h"
$runtimeLib = Join-Path $resolvedRuntimeDir ("libs\" + $pythonLibName)

if ((Test-Path $runtimeHeader) -and (Test-Path $runtimeLib)) {
    Write-Host ("Python dev files already present for " + $resolvedRuntimeDir)
    exit 0
}

$donorRuntime = Find-LocalDevDonor -RepoRootPath $resolvedRepoRoot -TargetRuntimeDir $resolvedRuntimeDir -PythonLibName $pythonLibName
if ($donorRuntime) {
    Write-Host ("Using local donor runtime dev files from " + $donorRuntime)
    Copy-DevFilesFromDonor -DonorRuntimeDir $donorRuntime -TargetRuntimeDir $resolvedRuntimeDir -PythonLibName $pythonLibName
    Write-Host ("Provisioned Python dev files into " + $resolvedRuntimeDir)
    exit 0
}

Copy-DevFilesFromNuGet -RepoRootPath $resolvedRepoRoot -TargetRuntimeDir $resolvedRuntimeDir -Version ([string]$versionInfo.version) -PythonLibName $pythonLibName
Write-Host ("Provisioned Python dev files into " + $resolvedRuntimeDir)
