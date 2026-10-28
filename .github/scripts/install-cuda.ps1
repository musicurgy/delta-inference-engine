# install-cuda.ps1 - the parts of the CUDA Toolkit 13.0.2 the engine's build needs, from NVIDIA's redistributable
# archives (https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json), into one folder laid
# out like an installed toolkit (bin\, include\, lib\x64\, nvvm\).  Each archive is checked against the SHA-256 pinned
# below (copied from that JSON) before it is unpacked.  No installer, no Visual Studio integration: build-ds41.bat
# builds with Ninja, which needs only nvcc and the libraries.
#   install-cuda.ps1 -Dest <folder>     then point CUDA_PATH at <folder> (on a GitHub runner it is set for later steps)
param([Parameter(Mandatory = $true)][string]$Dest)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Windows PowerShell's progress bar slows Invoke-WebRequest down many times
$Base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
$Parts = @(
  @{ Path = 'cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-13.0.88-archive.zip';
     Sha = '61760e24a44937c6c29a20ddea69067b4d3860ccd6a9a0564573f840d96e6c5f' },
  @{ Path = 'cuda_crt/windows-x86_64/cuda_crt-windows-x86_64-13.0.88-archive.zip';
     Sha = 'ff413c89dbd96ed3b3dcfac923437de78acea32fb90309277197c8a127588112' },
  @{ Path = 'libnvvm/windows-x86_64/libnvvm-windows-x86_64-13.0.88-archive.zip';
     Sha = '6dda4a82d22a2c173a65c66b4b4c933e3424a56ce845e2864bdca9cc66e6c524' },
  @{ Path = 'cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.0.96-archive.zip';
     Sha = 'a2ed875f9997aa24904fb70cc9db3acd9308433cde99bc8e63ec1271c9da31b4' },
  @{ Path = 'libcublas/windows-x86_64/libcublas-windows-x86_64-13.1.0.3-archive.zip';
     Sha = '4ac4847bbe4f7709b244956fcfc32197a2954ee70b155cb67eebd9ee26f7e339' },
  @{ Path = 'cuda_cccl/windows-x86_64/cuda_cccl-windows-x86_64-13.0.85-archive.zip';
     Sha = 'f4f5187f7976d8057d76e1e04e72c8b7d0a2a22793107a671ad709b1d8d3408d' },
  @{ Path = 'cuda_profiler_api/windows-x86_64/cuda_profiler_api-windows-x86_64-13.0.85-archive.zip';
     Sha = '38c6c854f325b64e7ca0a8b14e1996dedb5abbb38f3580e5c7744b688abdc56e' }
)
$Dest = [IO.Path]::GetFullPath($Dest)
$Work = Join-Path $Dest '_download'
New-Item -ItemType Directory -Force $Work | Out-Null
foreach ($p in $Parts) {
  $name = Split-Path $p.Path -Leaf
  $zip = Join-Path $Work $name
  if (-not (Test-Path $zip) -or (Get-FileHash $zip -Algorithm SHA256).Hash -ne $p.Sha) {
    Write-Host "downloading $name"
    Invoke-WebRequest -Uri ($Base + $p.Path) -OutFile $zip -UseBasicParsing
  }
  $got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
  if ($got -ne $p.Sha) { throw "$name has SHA-256 $got, not the pinned $($p.Sha)" }
  # each archive holds one folder (<name without .zip>\bin, \include, \lib ...): its contents merge into $Dest
  # (Windows' own tar.exe, bsdtar, unpacks zips; Git's GNU tar, which may come first on PATH, does not)
  & "$env:SystemRoot\System32\tar.exe" -xf $zip -C $Dest --strip-components=1 --exclude '*/LICENSE'
  if ($LASTEXITCODE -ne 0) { throw "could not unpack $name" }
  Write-Host "  $name OK (sha256 $got)"
}
Remove-Item -Recurse -Force $Work
& (Join-Path $Dest 'bin\nvcc.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'nvcc does not run' }
if ($env:GITHUB_ENV) {
  "CUDA_PATH=$Dest" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
  Join-Path $Dest 'bin' | Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
}
