# Configuration
#$LocalDir  = "C:\Users\user\source\repos\mzp-images-training-processing\FTS_JPG_Images_Classification\RAW_FTS_GOOD_FAINT"
$LocalDir = Join-Path $PSScriptRoot "RAW_FTS_GOOD_FAINT"
$RemoteUser = "nikolair"
$RemoteHost = "132.75.xxx.yy"
$RemoteDir  = "/lustre1/home/HPC_CALC/FACULTY_STORAGE/ASTRONOMICAL_STORAGE/DATA_MIZPE_IMAGES"

# Upload directory recursively
scp -r "$LocalDir" "${RemoteUser}@${RemoteHost}:${RemoteDir}"

# Check result
if ($LASTEXITCODE -eq 0) {
    Write-Host "Upload completed successfully!" -ForegroundColor Green
}
else {
    Write-Host "Upload failed!" -ForegroundColor Red
}