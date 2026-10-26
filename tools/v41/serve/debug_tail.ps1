# debug_tail.ps1 - shows a growing file in this console as it is written, character by character (so a prompt such as
# "Download it now [Y,N]?" shows before its line ends), until the -Done file appears; then prints the rest and stops.
# debug_run.bat runs the launcher with its output going to the file and this beside it in the same window.
param([Parameter(Mandatory = $true)][string]$File, [Parameter(Mandatory = $true)][string]$Done)
while (-not (Test-Path $File)) { Start-Sleep -Milliseconds 100 }
$s = [IO.File]::Open($File, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
$r = New-Object IO.StreamReader($s, [Text.Encoding]::Default)
$buf = New-Object char[] 4096
while ($true) {
    $n = $r.Read($buf, 0, $buf.Length)
    if ($n -gt 0) { [Console]::Out.Write($buf, 0, $n); continue }
    if (Test-Path $Done) {   # the launcher has ended: what is left, then stop
        while (($n = $r.Read($buf, 0, $buf.Length)) -gt 0) { [Console]::Out.Write($buf, 0, $n) }
        break
    }
    Start-Sleep -Milliseconds 100
}
$r.Close()
