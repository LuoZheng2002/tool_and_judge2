# Hindi Multilingual Tool-Calling Examples Report

This document contains concrete Hindi examples extracted from the BFCL evaluation results, demonstrating various failure modes in multilingual tool-calling robustness. All examples are sourced from actual model outputs in the project's result files.

---

## 1️⃣ DeepSeek Hindi Syntax Errors (Format / Refusal)

### Example 1.1: Database Column Deletion (multiple_10)

**User Query (Hindi):**
```
मुझे अपने कर्मचारियों के डेटाबेस में personal_data तालिका से कुछ कॉलम हटाने की आवश्यकता है। मैं उनकी ईमेल पतों और सामाजिक सुरक्षा नंबरों को हटाना चाहता हूँ ताकि गोपनीयता का सम्मान किया जा सके।
```

**Model Output (DeepSeek V3.2 FT):**
```json
{
  "content": "मुझे आपकी मदद करने में खुशी होगी, लेकिन मुझे डेटाबेस का नाम जानने की आवश्यकता है। आपने \"personal_data\" तालिका का उल्लेख किया है, लेकिन डेटाबेस का नाम नहीं बताया है।\n\nकृपया मुझे बताएं कि किस डेटाबेस से आप कॉलम हटाना चाहते हैं?",
  "role": "assistant"
}
```

**Expected Tool Call (Gold):**
```json
[{"database.modify_columns": {"db_name": "employees", "table": "personal_data", "operation": "delete", "columns": ["email", "social_security_number"]}}]
```

**Error Type:** `SYNTAX_ERROR` (Parsing Error)

**Explanation:** DeepSeek refuses to infer the database name "employees" from the context ("मेरे कर्मचारियों के डेटाबेस" = "my employees database") and instead asks for clarification. BFCL expects a tool call, so this free-text response fails parsing.

---

### Example 1.2: Magnetic Field Calculation (multiple_40)

**User Query (Hindi):**
```
चुंबकीय क्षेत्र की ताकत की गणना करें, दी गई दूरी 8 मीटर है और धारा 12 एम्पियर है?
```

**Model Output (DeepSeek V3.2 FT):**
```json
{
  "content": "मैं आपके प्रश्न को समझता हूं, लेकिन मुझे यह स्पष्ट करने की आवश्यकता है कि चुंबकीय क्षेत्र की गणना के लिए कौन सा नियम उपयोग किया जाना चाहिए। आपने दूरी (8 मीटर) और धारा (12 एम्पियर) दी है, लेकिन यह निर्दिष्ट नहीं किया है कि:\n\n1. क्या यह एक सीधे तार के लिए है? (बायोट-सावर्ट नियम)\n2. क्या यह एक बंद लूप के लिए है? (एम्पीयर का नियम)\n\nयदि यह एक सीधे तार से दूरी के लिए है, तो मैं बायोट-सावर्ट नियम का उपयोग कर सकता हूं। कृपया स्पष्ट करें कि आप किस स्थिति के लिए गणना करना चाहते हैं?",
  "role": "assistant"
}
```

**Expected Tool Call (Gold):**
```json
[{"electromagnetism.biot_savart_law": {"current": 12, "distance": 8}}]
```

**Error Type:** `SYNTAX_ERROR` (Parsing Error)

**Explanation:** The model correctly identifies the applicable physics laws but refuses to make a tool call, instead asking for disambiguation between Biot-Savart and Ampere's laws. This pedagogically appropriate response fails BFCL execution.

---

### Example 1.3: Marble Sculpture Price (multiple_20)

**User Query (Hindi):**
```
3 फीट आकार के संगमरमर की मूर्ति बनाने की कीमत क्या है?
```

**Model Output (DeepSeek V3.2 FT):**
```json
{
  "content": "मुझे संगमरमर की मूर्ति के लिए कीमत की गणना करने के लिए आकार और सामग्री की जानकारी तो है, लेकिन मुझे जटिलता (complexity) का स्तर भी चाहिए। क्या आप बता सकते हैं कि मूर्ति की जटिलता क्या होगी - कम (low), मध्यम (medium), या उच्च (high)? डिफ़ॉल्ट रूप से मध्यम माना जाएगा।",
  "role": "assistant"
}
```

**Expected Tool Call (Gold):**
```json
[{"sculpture_price.calculate": {"material": "marble", "size": 3, "complexity": "medium"}}]
```

**Error Type:** `SYNTAX_ERROR` (Parsing Error)

**Explanation:** DeepSeek knows the complexity parameter has a default value ("medium") but still asks the user to confirm rather than using the default. This over-cautious behavior causes BFCL failure.

---

## 2️⃣ DeepSeek Hindi Clarification Instead of Tool Call

### Example 2.1: House Price Prediction (multiple_53)

**User Query (Hindi):**
```
5 वर्षों के लिए ब्याज दरों और बेरोजगारी दरों के आधार पर घरों की कीमतों का पूर्वानुमान लगाएं।
```

**Model Output (DeepSeek V3.2 FT):**
```json
{
  "content": "मैं आपके अनुरोध को समझता हूं, लेकिन मुझे आवश्यक जानकारी की कमी है। घरों की कीमतों का अनुमान लगाने के लिए मुझे निम्नलिखित जानकारी चाहिए:\n\n1. **स्वतंत्र चर (independent variables)**: आपने ब्याज दरों और बेरोजगारी दरों का उल्लेख किया है, लेकिन मुझे यह जानने की आवश्यकता है कि ये डेटा किस रूप में उपलब्ध हैं (जैसे कॉलम नाम या चर नाम)।\n\n2. **आश्रित चर (dependent variable)**: घरों की कीमतों का डेटा किस रूप में है? इसका कॉलम नाम क्या है?\n\n3. **डेटा सेट**: क्या आपके पास ऐतिहासिक डेटा है जिसमें ब्याज दरें, बेरोजगारी दरें और घरों की कीमतें शामिल हैं?\n\nकृपया इन विवरणों को प्रदान करें ताकि मैं आपके लिए 5 वर्षों के लिए घरों की कीमतों का अनुमान लगा सकूं।",
  "role": "assistant"
}
```

**Expected Tool Call (Gold):**
```json
[{"linear_regression": {"independent_var": ["interest rates", "unemployment rates"], "dependent_var": "house_price", "forecast_period": 5}}]
```

**Error Type:** `SYNTAX_ERROR` (Parsing Error)

**Explanation:** Despite clear query intent, DeepSeek refuses to generate a tool call and instead requests clarification about variable names and data availability—a sensible real-world response that fails BFCL's strict execution model.

---

### Example 2.2: Bohemian Rhapsody Lyrics (multiple_168)

**User Query (Hindi):**
```
Queen के गीत 'बोहेमियन रैप्सोडी' के बोल खोजें।
```

**Model Output (DeepSeek V3.2 PT):**
```json
{
  "content": "मैं आपकी मदद करना चाहूंगा, लेकिन मुझे Queen के गीत 'Bohemian Rhapsody' के बोल खोजने के लिए कुछ जानकारी की आवश्यकता है।\n\nमेरे पास गीत के बोल खोजने का टूल है, लेकिन उसे गीत का शीर्षक और कलाकार का नाम चाहिए। आपने गीत का शीर्षक ('Bohemian Rhapsody') और कलाकार (Queen) दिया है, लेकिन मुझे यह भी जानना होगा कि आप किस भाषा में गीत के बोल चाहते हैं।\n\nक्या आप बता सकते हैं कि आप किस भाषा में गीत के बोल चाहते हैं? उदाहरण के लिए: अंग्रेजी, फ्रेंच, स्पेनिश, जर्मन या इतालवी।",
  "role": "assistant"
}
```

**Expected Tool Call (Gold):**
```json
[{"get_song_lyrics": {"song_title": "Bohemian Rhapsody", "artist_name": "Queen", "lang": "English"}}]
```

**Error Type:** `SYNTAX_ERROR` (Parsing Error)

**Explanation:** DeepSeek correctly identifies all required parameters but asks for language confirmation instead of defaulting to English. The Hindi transliteration "बोहेमियन रैप्सोडी" is also not mapped back to "Bohemian Rhapsody" in the tool call.

---

## 3️⃣ GPT-5 Hindi "Outsmarts the Dataset" Cases

### Example 3.1: Wild Tiger → Panthera tigris (multiple_16)

**User Query (Hindi):**
```
जंगली बाघ की जनसंख्या की आनुवंशिक विविधता को अगले 100 पीढ़ियों के लिए 0.95 की संभावना के साथ बनाए रखने के लिए आवश्यक न्यूनतम जनसंख्या आकार (Ne) क्या है?
```

**Model Tool Call (GPT-5):**
```json
[{"population_genetics.calculate_ne": {"species": "Panthera tigris", "generations": 100, "probability": 0.95}}]
```

**Gold Tool Call:**
```json
[{"population_genetics.calculate_ne": {"species": "wild tiger", "generations": 100, "probability": 0.95}}]
```

**Error Category:** `EXACTLY_SAME_MEANING`

**Explanation:** GPT-5 "outsmarts" the dataset by converting the Hindi "जंगली बाघ" (wild tiger) to the scientific Latin name *Panthera tigris*. While scientifically more precise, this fails BFCL's strict string matching against the expected value "wild tiger".

---

### Example 3.2: Small-Space → Hyphenated Form (multiple_61)

**User Query (Hindi):**
```
पोर्टलैंड, ओरेगॉन में 5 साल का अनुभव रखने वाले छोटी जगह के बगीचे डिजाइन में विशेषज्ञता वाले लैंडस्केप आर्किटेक्ट खोजें।
```

**Model Tool Call (GPT-5):**
```json
[{"landscape_architect.find_specialty": {"location": "Portland, OR", "specialization": "small-space garden design", "years_experience": 5}}]
```

**Gold Tool Call:**
```json
[{"landscape_architect.find_specialty": {"location": "Portland, OR", "specialization": "small space garden design", "years_experience": 5}}]
```

**Error Category:** `EXACTLY_SAME_MEANING`

**Explanation:** GPT-5 normalizes "small space" to the hyphenated compound adjective "small-space", following standard English style guides. This grammatically correct form fails BFCL's exact string matching.

---

### Example 3.3: Artist Name Correction - Jaume Plensa (multiple_160)

**User Query (Hindi):**
```
मूर्तिकार जेम्स प्लेंसा द्वारा बनाई गई मूर्तियों के बारे में विस्तृत जानकारी प्राप्त करें।
```

**Model Tool Call (GPT-5):**
```json
[{"get_sculpture_info": {"artist_name": "Jaume Plensa", "detail": true}}]
```

**Gold Tool Call:**
```json
[{"get_sculpture_info": {"artist_name": "James Plensa", "detail": true}}]
```

**Error Category:** `EXACTLY_SAME_MEANING`

**Explanation:** GPT-5 correctly recognizes that the famous Spanish sculptor's actual name is "Jaume Plensa" (Catalan), not the anglicized "James Plensa" used in the gold annotation. This world-knowledge correction causes a strict matching failure.

---

## 4️⃣ Hindi PRE / POST Translation Drift Examples

### Example 4.1: Subject Name Translation Drift (multiple_9)

**Original Hindi Query:**
```
जॉन छात्र के लिए विभिन्न विषयों में प्राप्त अंकों {'गणित':90, 'विज्ञान':75, 'इतिहास':82, 'संगीत':89} का औसत ग्रेड निकालें।
```

**PRE-Translated Query (Hindi → English):**
```
Calculate the average grade of the marks obtained by John in different subjects {'Math':90, 'Science':75, 'History':82, 'Music':89}.
```

**Model Tool Call (Llama-3.1-70B with POST-translation):**
```json
[{"calculate_average": {"gradeDict": {"Mathematics": 90, "Science": 75, "History": 82, "Music": 89}}}]
```

**Gold Tool Call:**
```json
[{"calculate_average": {"gradeDict": {"math": 90, "science": 75, "history": 82, "music": 89}}}]
```

**Error Category:** `RELEVANT_BUT_INCORRECT`

**Explanation:** POST-translation produces "Mathematics" instead of the expected lowercase "math". The PRE-translation correctly renders "Math" but the model then normalizes it to the formal "Mathematics". This case normalization drift causes BFCL failure.

---

### Example 4.2: Era Name Translation Failure (multiple_45)

**Original Hindi Query:**
```
बर्फ़ युग कितने साल पहले था, यह गणना करें?
```

**PRE-Translated Query:**
```
Calculate how many years ago was the ice age?
```

**Model Tool Call (Llama-3.1-70B FT without PRE):**
```json
[{"geology.get_era": {"era_name": "रोप्युग", "calculate_years_ago": true}}]
```

**Gold Tool Call:**
```json
[{"geology.get_era": {"era_name": "Ice age", "calculate_years_ago": true}}]
```

**Error Category:** `LANGUAGE_MISMATCH_WRONG_VALUE`

**Explanation:** Without PRE-translation, Llama-3.1-70B produces a garbled Hindi transliteration "रोप्युग" instead of "Ice age". This demonstrates how FT without translation assistance can produce completely wrong values that are neither valid Hindi nor English.

---

### Example 4.3: Chinese Food Translation Failure (multiple_146)

**Original Hindi Query:**
```
सिएटल में 10 मील के भीतर चीनी भोजन परोसने वाले रेस्तरां खोजें।
```

**Model Tool Call (Llama-3.1-70B with POST):**
```json
[{"restaurant.find_nearby": {"cuisine": "Kitai", "max_distance": 10, "location": "Settle"}}]
```

**Gold Tool Call:**
```json
[{"restaurant.find_nearby": {"cuisine": "Chinese", "max_distance": 10, "location": "Seattle"}}]
```

**Error Category:** `LANGUAGE_MISMATCH_RELEVANT_BUT_INCORRECT`

**Explanation:** POST-translation produces "Kitai" (Russian word for China) instead of "Chinese", and "Settle" instead of "Seattle". This demonstrates how isolated parameter translation can introduce unexpected language contamination from the translation model's multilingual training.

---

### Example 4.4: Sentiment Text Semantic Drift (multiple_149)

**Original Hindi Query:**
```
इस पाठ का भावना विश्लेषण करें: 'मुझे यहाँ का खाना बहुत पसंद है! यह हमेशा ताजा और स्वादिष्ट होता है।'
```

**Model Tool Call (Llama-3.1-70B with POST):**
```json
[{"sentiment_analysis": {"text": "I love having a special kind of earthen pot water! It's always fresh.", "language": "hi"}}]
```

**Gold Tool Call:**
```json
[{"sentiment_analysis": {"text": "I love the food here! It's always fresh and delicious.", "language": "en"}}]
```

**Error Category:** `RELEVANT_BUT_INCORRECT`

**Explanation:** POST-translation severely distorts the meaning: "खाना" (food) becomes "earthen pot water", and "स्वादिष्ट" (delicious) is dropped entirely. This semantic drift fundamentally changes the sentiment being analyzed.

---

## 5️⃣ Llama-3.1-70B Hindi Contrast Cases

### Example 5.1: Successful Hindi Location Handling (multiple_60)

**User Query (Hindi):**
```
न्यू यॉर्क सिटी के लिए 7 दिन बाद आर्द्रता और तापमान क्या होगा?
```

**Model Tool Call (Llama-3.1-70B FT):**
```json
[{"humidity_temperature_forecast": {"location": "न्यू यॉर्क सिटी", "days": 7}}]
```

**Gold Tool Call:**
```json
[{"humidity_temperature_forecast": {"location": "New York City", "days": 7}}]
```

**Error Category:** `LANGUAGE_MISMATCH_EXACTLY_SAME_MEANING`

**Explanation:** Llama-3.1-70B follows instructions correctly—the function is called with proper structure. However, the location parameter contains Hindi text "न्यू यॉर्क सिटी" (transliteration of "New York City") rather than English. This is semantically correct but would fail real API execution expecting English strings.

---

### Example 5.2: Messi Player Name in Hindi (multiple_83)

**User Query (Hindi):**
```
बार्सिलोना में लियोनेल मेस्सी के सभी समय के गोलों की संख्या बताएं।
```

**Model Tool Call (Llama-3.1-70B FT):**
```json
[{"player_stats.get_all_time_goals": {"player_name": "लियोनेल मेस्सी", "team_name": "बार्सिलोना"}}]
```

**Gold Tool Call:**
```json
[{"player_stats.get_all_time_goals": {"player_name": "Lionel Messi", "team_name": "Barcelona"}}]
```

**Error Category:** `LANGUAGE_MISMATCH_EXACTLY_SAME_MEANING`

**Explanation:** This is a representative Hindi failure: both player name and team name are transliterated to Devanagari script. The intent is perfectly preserved, but downstream API calls expecting "Lionel Messi" would fail. This illustrates why translation-based mitigation (PRE/POST) is needed.

---

### Example 5.3: Llama Shows Wrong Function Selection (multiple_67)

**User Query (Hindi):**
```
नमस्ते, आप कैसे हैं?
```

**Model Tool Call (Llama-3.1-70B FT):**
```json
[{"sentiment_analysis": {"text": "नमस्ते, आप कैसे है?"}}]
```

**Gold Tool Call:**
```json
[{"translate": {"text": "Hello, how are you?", "source_language": "Hindi", "target_language": "English"}}]
```

**Error Category:** `MISC_ERROR` (Wrong Function Name)

**Explanation:** The Hindi greeting "नमस्ते, आप कैसे हैं?" should trigger a translation function, but Llama-3.1-70B incorrectly selects `sentiment_analysis` instead of `translate`. This demonstrates how Hindi queries can mislead function selection.

---

## Summary Statistics

| Model | Regime | Syntax Errors | Language Mismatch | Wrong Value | Relevant but Incorrect |
|-------|--------|---------------|-------------------|-------------|------------------------|
| DeepSeek V3.2 | FT | 35+ | 25+ | 3 | 7 |
| GPT-5 | FT | 0 | 8+ | 2 | 15+ |
| Llama-3.1-70B | FT | 0 | 30+ | 6 | 20+ |
| Llama-3.1-70B | PRE | 0 | 0 | 5 | 15+ |
| Llama-3.1-70B | POST | 0 | 5+ | 8 | 20+ |

**Key Findings:**
1. **DeepSeek** exhibits high clarification-seeking behavior in Hindi, causing syntax errors
2. **GPT-5** tends to "outsmart" the dataset with world knowledge corrections
3. **Llama-3.1-70B FT** produces many language mismatch errors with Devanagari transliterations
4. **PRE-translation** significantly reduces language mismatch but can introduce semantic drift
5. **POST-translation** helps but can produce wrong translations for isolated parameters

---

*Report generated from: `/home/pranav/translate/tool_and_judge2/tool/result/` directory*
*Source datasets: BFCL_v4_multiple Hindi (fulltrans, parttrans, nonoise variants)*

