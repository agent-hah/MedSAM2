Write-Host "Starting preprocessing for DIAS dataset..."
python run_preprocessing.py dias
if ($LASTEXITCODE -ne 0) { throw "DIAS preprocessing failed" }

Write-Host "Starting preprocessing for Rat dataset..."
python run_preprocessing.py rat
if ($LASTEXITCODE -ne 0) { throw "Rat preprocessing failed" }

Write-Host "Starting preprocessing for Human dataset..."
python run_preprocessing.py human
if ($LASTEXITCODE -ne 0) { throw "Human preprocessing failed" }

Write-Host "All datasets preprocessed successfully!"
