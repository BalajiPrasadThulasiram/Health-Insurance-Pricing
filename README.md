# Health Insurance Premium Pricing (GLM + Gradient Boosting)

**Dataset:** insurance.csv (Kaggle: Medical Cost Personal Datasets)

## Results (test)
- GLM — MAE: 7174.06, RMSE: 9990.77, R²: 0.357
- GBR — MAE: 2404.90, RMSE: 4328.15, R²: 0.879

## Run training
```bash
pip install -r requirements.txt
python train.py --csv data/insurance.csv --out outputs
```

## Optional API
```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @sample_request.json
```

## Notes
- GLM (Gamma/log) is interpretable baseline; GBR is more accurate.
- For production pricing: add plan design, deductibles/coinsurance, rating areas, prior utilization/claims, Rx, risk adjustment.
- Calibrate to target loss ratios and include admin/margin/risk load for actuarial soundness.
