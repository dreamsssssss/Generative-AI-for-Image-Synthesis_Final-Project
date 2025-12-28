# Generative-AI-for-Image-Synthesis_Final-Project

# Fine-tuning SDXL with DreamBooth + LoRA（個人化主體／風格客製化）

本專案以 **Stable Diffusion XL (SDXL Base 1.0)** 為基礎模型，整合 **DreamBooth**（主體綁定）與 **LoRA**（參數高效微調），在有限運算資源下完成「特定主體」的個人化文生圖微調與推論流程。  
我們以兩個案例做系統性實驗：真實主體 **Lilly cat** 與風格化角色 **Capoo**，比較不同訓練／推論步數對生成品質與主體一致性的影響，並分析風格化主體可能的 domain gap 問題。

> 期末報告：Generative AI for Image Synthesis – Fine-tuning Stable Diffusion XL with DreamBooth and LoRA

---

## 專案亮點
- **主體綁定（DreamBooth）**：以少量圖片將特定主體與獨特 token 綁定，提升生成一致性。
- **高效微調（LoRA）**：凍結大模型權重，只訓練輕量適配器，降低顯存與訓練成本。
- **自動化標註（BLIP captions）**：用 BLIP 生成 caption，並統一加上主體 token 前綴以強化關聯。
- **Colab/T4 可跑**：搭配 8-bit Adam、Gradient Checkpointing、fp16 等記憶體最佳化策略。

---

## 方法概述
- **Base Model**：Stable Diffusion XL（SDXL Base 1.0）
- **Subject-driven personalization**：DreamBooth（將「唯一 token」與主體視覺特徵強連結）
- **Parameter-efficient fine-tuning**：LoRA（僅更新低秩矩陣，權重檔更小、訓練更省資源）
- **部署**：輸出 LoRA 權重為 `.safetensors`，可上傳 Hugging Face Hub 做版本管理，推論時動態載入 LoRA。

---

## 流程
本專案流程分為三階段：

1. **Data Preparation**
   - 影像收集與預處理（過濾格式、移除不相關檔案）
   - **BLIP 自動 caption**
   - 統一在 caption 前加入概念 token 前綴（如：`a photo of Lilly cat, ...`）

2. **SDXL Training（DreamBooth + LoRA）**
   - 使用 LoRA 注入 cross-attention 等層，只訓練少量參數
   - 記憶體最佳化：8-bit Adam、Gradient Checkpointing、Mixed Precision (fp16)
   - 訓練配置：1024×1024、常數學習率、SNR Gamma 等

3. **Model Storage & Inference**
   - 將 LoRA 權重封裝為 `.safetensors`
   - 推論時載入 SDXL Base + VAE，並 `load_lora_weights` 動態套用 LoRA
   - 輸入含 token 的 prompt 生成個人化影像

---

## 實驗設定
- 平台：Google Colab（NVIDIA T4 GPU）
- 基礎模型：SDXL Base 1.0
- 解析度：1024×1024
- 訓練步數（示範設定）：50
- 學習率：1e-4
- 優化器：8-bit Adam

資料集：
- **Lilly cat**：15 張真實貓咪照片，caption 前綴：`a photo of Lilly cat`
- **Capoo**：15 張二維平面角色圖，caption 前綴：`a photo of capoo`

---


