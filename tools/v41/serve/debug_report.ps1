# debug_report.ps1 - what "Start DeepSeek (debug).bat" collects for a bug report (read only: it changes nothing).
#
#   debug_report.ps1 -Dir <report folder> -Phase before   the machine and the install, a quick PASS / FAIL list
#   debug_report.ps1 -Dir <report folder> -Phase after    the run's logs, crash records; the folder zipped
#
# Never collected: prompts, answers, chat history, files outside this folder and the model folder's listing, keys.
# The Windows user name and profile path are replaced by <user> / <profile> in everything that goes into the zip.
param([Parameter(Mandatory = $true)][string]$Dir, [ValidateSet("before", "after")][string]$Phase = "before")

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null

function Section([string]$file, [string]$title, [scriptblock]$body) {
    $out = Join-Path $Dir $file
    Add-Content -Path $out -Value "`r`n===== $title =====" -Encoding UTF8
    try { & $body 2>&1 | Out-String -Width 220 | Add-Content -Path $out -Encoding UTF8 }
    catch { Add-Content -Path $out -Value "(failed: $($_.Exception.Message))" -Encoding UTF8 }
}

function Check([string]$what, [bool]$ok, [string]$detail) {
    $line = "{0,-5} {1}{2}" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $what, $(if ($detail) { " - $detail" } else { "" })
    Add-Content -Path (Join-Path $Dir "checks.txt") -Value $line -Encoding UTF8
}

function Settings() {
    # ds41-paths.bat's settings as the launcher sees them (DS41_MODELS, DS41_GPU, ...)
    $lines = cmd /c "call `"$Root\ds41-paths.bat`" >nul 2>&1 & set DS41_" 2>$null
    $h = @{}
    foreach ($l in $lines) { if ($l -match "^(DS41_[A-Z_]+)=(.*)$") { $h[$matches[1]] = $matches[2] } }
    return $h
}

if ($Phase -eq "before") {
    Set-Content -Path (Join-Path $Dir "started.txt") -Value (Get-Date -Format o) -Encoding UTF8
    Section "system.txt" "Delta Inference Engine" {
        "folder: $Root  (path length $($Root.Length))"
        if (Test-Path (Join-Path $Root ".git")) {
            "git: $(git -C $Root log -1 --format='%h %ad %s' --date=short 2>&1)"
            "branch: $(git -C $Root rev-parse --abbrev-ref HEAD 2>&1); changed files: $((git -C $Root status --porcelain --untracked-files=no 2>$null | Measure-Object).Count)"
        } else { "not a git clone (a ZIP download)" }
        if (Test-Path (Join-Path $Root "model-choice.txt")) { "model-choice.txt: $(Get-Content (Join-Path $Root 'model-choice.txt') -TotalCount 1)" }
        else { "model-choice.txt: none (the default model)" }
    }
    Section "system.txt" "Windows" {
        $os = Get-CimInstance Win32_OperatingSystem
        "$($os.Caption) $($os.Version) build $($os.BuildNumber), $($os.OSArchitecture); up since $($os.LastBootUpTime)"
        "RAM: {0:N1} GiB installed, {1:N1} GiB free now" -f ($os.TotalVisibleMemorySize / 1MB), ($os.FreePhysicalMemory / 1MB)
        "page file: $((Get-CimInstance Win32_PageFileUsage | ForEach-Object { '{0} {1} MB' -f $_.Name, $_.AllocatedBaseSize }) -join ', ')"
        Get-CimInstance Win32_PhysicalMemory | ForEach-Object { "  RAM stick: {0:N0} GB, rated {1} MT/s, running {2} MT/s" -f ($_.Capacity / 1GB), $_.Speed, $_.ConfiguredClockSpeed }
    }
    Section "system.txt" "CPU" {
        Get-CimInstance Win32_Processor | ForEach-Object { "$($_.Name) - $($_.NumberOfCores) cores, $($_.NumberOfLogicalProcessors) threads" }
        if (Test-Path $Py) {
            "instruction level: $(& $Py (Join-Path $Root 'tools\v41\serve\cpu_level.py') 2>&1)"
            "engine build it picks: $(& $Py (Join-Path $Root 'tools\v41\serve\cpu_level.py') --pick-build 2>&1)"
        }
    }
    Section "system.txt" "NVIDIA" {
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,pcie.link.gen.current,pcie.link.width.current,power.limit,display_active --format=csv
        nvidia-smi
    }
    Section "system.txt" "Drives" {
        Get-PhysicalDisk | Format-Table FriendlyName, MediaType, BusType, @{n = "GB"; e = { [math]::Round($_.Size / 1e9) } } -AutoSize
        Get-Volume | Where-Object DriveLetter | Format-Table DriveLetter, FileSystem, @{n = "free GB"; e = { [math]::Round($_.SizeRemaining / 1e9) } }, @{n = "size GB"; e = { [math]::Round($_.Size / 1e9) } } -AutoSize
    }
    Section "system.txt" "Python" {
        "py launcher: $((py -0 2>&1) -join ' | ')"
        if (Test-Path $Py) {
            "venv: $(& $Py --version 2>&1)"
            & $Py -m pip list --disable-pip-version-check 2>&1 | Select-String -Pattern "huggingface|tokenizers|numpy|requests|jinja"
        } else { "no .venv yet" }
    }
    Section "system.txt" "Windows settings for speed (read only)" {
        if (Test-Path $Py) { & $Py (Join-Path $Root "tools\v41\serve\windows_tips.py") 2>&1 }
        "power plan: $((powercfg /getactivescheme) -join ' ')"
    }
    Section "system.txt" "Engine builds" {
        foreach ($b in "build-ds41", "build-ds41-portable", "build-ds41-compat-avx", "build-ds41-compat") {
            $d = Join-Path $Root $b
            if (-not (Test-Path $d)) { continue }
            $exe = Join-Path $d "ds41_generate.exe"
            $lvl = if (Test-Path (Join-Path $d "cpu-level.txt")) { Get-Content (Join-Path $d "cpu-level.txt") -TotalCount 1 } else { "?" }
            "$b : needs $lvl; files: $((Get-ChildItem $d -File -Include *.exe, *.dll -Name -Recurse -Depth 0) -join ', ')"
            if (Test-Path $exe) {
                $p = Start-Process -FilePath $exe -NoNewWindow -Wait -PassThru -RedirectStandardError (Join-Path $Dir "exe-$b.txt") -RedirectStandardOutput (Join-Path $Dir "exe-$b-out.txt")
                "  ds41_generate.exe with no arguments: exit code $($p.ExitCode) (2 = it loads on this PC and prints its usage; -1073741795 = an instruction this CPU lacks; -1073741515 = a DLL missing)"
            }
        }
    }
    $s = Settings
    $models = $s["DS41_MODELS"]
    Section "system.txt" "Settings (ds41-paths.bat)" { $s.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" } }
    Section "system.txt" "Model folder" {
        if ($models -and (Test-Path $models)) {
            "$models"
            Get-ChildItem $models -Recurse -Depth 2 -File -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch "\\\.cache\\" } |
                Format-Table @{n = "file"; e = { $_.FullName.Substring($models.Length) } }, @{n = "GB"; e = { [math]::Round($_.Length / 1e9, 2) } }, LastWriteTime -AutoSize
        } else { "no model folder at '$models' yet" }
    }
    if (Test-Path $Py) {
        Section "system.txt" "The launcher's config check (ds41_local.py)" {
            & $Py (Join-Path $Root "tools\v41\serve\ds41_local.py") (Join-Path $Root "tools\v41\serve\ds41-windows.json") (Join-Path $Dir "ds41-windows.local.json") 2>&1
        }
    }
    # the quick list: what a reader checks first
    $drv = (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    Check "NVIDIA driver 580 or newer" ($drv -and [int]($drv.Split(".")[0]) -ge 580) "$drv"
    $vram = (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | ForEach-Object { [int]$_ } | Measure-Object -Maximum).Maximum
    Check "a GPU with 12 GB+ of VRAM" ($vram -ge 11800) "$vram MiB"
    $ram = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
    Check "128 GB+ of RAM" ($ram -ge 120) ("{0:N0} GiB" -f $ram)
    Check "Python environment (.venv)" (Test-Path $Py) ""
    $eng = $false
    foreach ($b in "build-ds41", "build-ds41-portable", "build-ds41-compat-avx", "build-ds41-compat") { if (Test-Path (Join-Path $Root "$b\ds41_generate.exe")) { $eng = $true } }
    Check "an engine build" $eng ""
    Check "the folder path is short (under 60 characters)" ($Root.Length -lt 60) "$($Root.Length)"
    if ($models -and (Test-Path $models)) {
        $free = (Get-PSDrive -Name $models.Substring(0, 1) -ErrorAction SilentlyContinue).Free / 1e9
        Check "the model folder exists" $true "$models, $([math]::Round($free)) GB free"
    } else { Check "the model folder exists" $false "$models" }
    if (Test-Path $Py) {
        & $Py (Join-Path $Root "tools\v41\serve\ds41_local.py") (Join-Path $Root "tools\v41\serve\ds41-windows.json") (Join-Path $Dir "x.json") *> $null
        Check "every model file the launcher needs is there" ($LASTEXITCODE -eq 0) "see system.txt, 'The launcher's config check'"
        Remove-Item (Join-Path $Dir "x.json") -ErrorAction SilentlyContinue
    }
    Write-Host "Debug report: the machine and the install checked (checks.txt)."
    Get-Content (Join-Path $Dir "checks.txt") | ForEach-Object { Write-Host "  $_" }
    exit 0
}

# ---- after the run: its logs, crash records, then everything scrubbed and zipped
$started = [datetime](Get-Content (Join-Path $Dir "started.txt") -TotalCount 1)
$log = Join-Path $Root "logs\strata-ds41.log"
if (Test-Path $log) {   # the engine / server log: this run's part (it is appended to), at most 20,000 lines
    Get-Content $log -Tail 20000 | Set-Content (Join-Path $Dir "strata-ds41.log.txt") -Encoding UTF8
}
$cfg = Join-Path $Root "tools\v41\serve\local\ds41-windows.json"
if (Test-Path $cfg) { Copy-Item $cfg (Join-Path $Dir "ds41-windows.run.json") }
Section "crashes.txt" "Windows Application log: errors since the run started (crashes, missing DLLs)" {
    Get-WinEvent -FilterHashtable @{ LogName = "Application"; Level = 1, 2; StartTime = $started } -ErrorAction SilentlyContinue |
        Where-Object { $_.Message -match "ds41_generate|python|nvcuda|cudart|cublas" } |
        Select-Object -First 30 | Format-List TimeCreated, ProviderName, Id, Message
}
Section "crashes.txt" "NVIDIA after the run" { nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu,clocks_throttle_reasons.active --format=csv }

# scrub: the Windows user name and profile path out of every text file
$user = $env:USERNAME
$profilePath = $env:USERPROFILE
Get-ChildItem $Dir -File | ForEach-Object {
    $t = Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue
    if ($null -eq $t) { return }
    if ($profilePath) { $t = $t.Replace($profilePath, "<profile>").Replace($profilePath.Replace("\", "/"), "<profile>") }
    if ($user -and $user.Length -ge 3) { $t = [regex]::Replace($t, [regex]::Escape($user), "<user>", "IgnoreCase") }
    Set-Content -Path $_.FullName -Value $t -Encoding UTF8 -NoNewline
}
$zip = "$Dir.zip"
Compress-Archive -Path (Join-Path $Dir "*") -DestinationPath $zip -Force
Write-Host ""
Write-Host "Debug report ready: $zip"
Write-Host "Send that file with a short note of what went wrong.  It holds no prompts or answers; your Windows user name"
Write-Host "is replaced by <user>.  You can open it to check before you send it."
if (-not $env:DS41_DEBUG_QUIET) { Start-Process explorer.exe -ArgumentList "/select,`"$zip`"" }   # (tests: no window)
exit 0
