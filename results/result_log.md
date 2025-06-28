# 2025_0628_2 - Implemented label smoothing to solve overfitting
![](Attention_Decoder_2025_0628_2.png)
--- 
- Added label smoothing to solve overfitting.
- Model not overfitting at the first ~10 epochs but overfitting emerge afterward.
- Model still tends to use frequent token to lower loss.
- Possible solution 1: Apply class weighting to frequent tokens.
- Possible solution 2: Apply substring tokenizer to preserve semantic of rare tokens.
---

# 2025_0628 - Implemented techniques to solve overfitting
![](Attention_Decoder_2025_0628.png)
---
- Added weight decay.
- Implemented teacher forcing ratio.
- Adjusted dropout percentage and learning rate.
- Still observed overfitting.
---

# 2025_0627 - Seq2Seq on Code to Documentation Dataset
![](Attention_Decoder_2025_0627.png)
---
- Experienced overfitting.
- Model tends to predict frequent token to lower train loss.
---

# 2025_0627 - Seq2Seq on French to English Dataset
![](Seq2Seq_Model_on_French2English_Dataset.png)
---
- Implemented sequence-to-sequence model with RNN encoder and Additive Attention+RNN decoder.
- Validated model performance on a simple english to french dataset.
---
