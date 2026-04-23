# KAN-Mobile Training Sprint Automation

# 1. Train Baseline
Write-Host "--- Starting Baseline Model Training ---" -ForegroundColor Cyan
python src/train.py --config configs/config_baseline.yaml

# 2. Train KAN
Write-Host "`n--- Starting FastKAN Model Training ---" -ForegroundColor Green
python src/train.py --config configs/config_kan.yaml

# 3. Final Summary
Write-Host "`n--- Sprint Complete! ---" -ForegroundColor Yellow
Write-Host "Results saved in logs/ directory."
Write-Host "Check your W&B Dashboard for the comparison charts."
