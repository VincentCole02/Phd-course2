# AT4｜EGE-UNet 短訓復現與離線證據核查

本目錄配合數據科學前沿分析研究 AT4 期末報告。[可編輯稿](paper.md)

## 分割實驗

使用 Python 3.11，從本目錄執行下列指令。可將課程 AT3 壓縮包中的 `data/isic2017/` 複製到本目錄，或執行資料準備程式取得公開縮圖和官方遮罩。

```bash
python -m pip install -r requirements.txt
python src/segmentation/prepare_data.py
python src/segmentation/run.py --device cpu --checkpoint results/segmentation/short-run/best.pt --output reruns/cpu-eval --no-plots
```

準備好的資料應包含 1,000 張訓練圖、150 張驗證圖和 600 張測試圖。預期 `manifest.csv` 的 SHA-256 為 `2c706ea2e88229df41b5d3be48dba722678ee2a0d56be7a83abce3c9ef33c30d`。若資料位於其他位置，可用 `--data PATH`。如需重新訓練，在支援 CUDA 的環境中執行以下指令。原始短訓使用 CUDA 12.1。

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
python src/segmentation/run.py --device cuda --epochs 30 --output reruns/training
```

每次執行請指定新的輸出目錄。按附帶權重與同一資料清單重評，600 張測試圖的 Dice 和前景 IoU 應在保留兩位小數後分別為 82.06% 和 69.58%。可用以下指令重算論文的逐圖統計、面積分層與影像 bootstrap 區間。

```bash
python src/segmentation/analyze_test.py results/segmentation/short-run/test_per_image.csv results/segmentation/test-analysis.json
```

[實驗設定與結果](results/segmentation/short-run/metrics.json) · [逐輪紀錄](results/segmentation/short-run/history.csv) · [逐圖紀錄](results/segmentation/short-run/test_per_image.csv) · [AT4 後分析](results/segmentation/test-analysis.json) · [AT4 CPU 重評](results/segmentation/cpu-check.json)

網路源碼與授權放在 `src/segmentation/vendor/`。[上游實作版本](https://github.com/JCruan519/EGE-UNet/tree/f52ba30c6bf7d0ca479c2c9d4a3cbda999f49d3a) · [ISIC2017 Task 1 資料與授權](https://challenge.isic-archive.com/data/#2017)

## 離線證據整理

此部分用 Python 標準庫執行。使用同一來源包分別跑 3 與 6 步組，然後用獨立腳本檢查簡報和日誌。

```bash
python src/agent/run_agent.py --output-dir reruns/agent
python src/agent/validate.py reruns/agent/max_steps_6
python src/agent/validate.py reruns/agent/max_steps_3
```

預期 6 步組在 5 次呼叫後通過。3 步組停在讀取階段，驗證腳本以退出碼 1 結束。Agent 的原始資料保留於 `results/agent/`。來源 ID、引用行號與工具回傳可用組別和步號連接。

[比較表](results/agent/comparison.csv) · [通過組簡報](results/agent/final-brief.md) · [評估表](results/agent/evaluation.csv) · [證據溯源](results/agent/provenance.json)
