# Research Question 1 — Explainability (MLR + SHAP)

**Subquestion:** What explainability results are produced by the proposed SHAP-guided MLR model in predicting the smishing class?

## Overview

SHAP (`LinearExplainer`) was applied to the trained Multinomial Logistic Regression model. The summaries below focus on the **smishing** class and cover mean absolute SHAP values, structured indicator contributions, influential TF-IDF terms, and local explanation examples.

## SHAP Run Metadata

- Background rows: 200
- Explained rows: 500
- Feature count: 5006
- Class labels: ham, smishing, spam

## URL, EMAIL, and PHONE Indicator Contributions

Mean absolute SHAP values for structured message-level features (target class: **smishing**):

| Feature | Mean Abs SHAP |
| --- | ---: |
| PHONE | 0.8386 |
| phone_count | 0.7141 |
| URL | 0.6296 |
| url_count | 0.2704 |
| EMAIL | 0.0184 |
| email_count | 0.0184 |

**Interpretation:** Higher values indicate stronger average influence on smishing predictions. PHONE, URL, and related count features typically rank among the most influential structured indicators in smishing messages.

## Top 15 Influential TF-IDF Terms (Smishing Class)

| Rank | Term | Mean Abs SHAP |
| ---: | --- | ---: |
| 1 | claim | 0.0681 |
| 2 | prize | 0.0403 |
| 3 | ingest | 0.0353 |
| 4 | customer | 0.0299 |
| 5 | embody | 0.0265 |
| 6 | phonetoken | 0.0263 |
| 7 | mobile | 0.0252 |
| 8 | contact | 0.0231 |
| 9 | message | 0.0213 |
| 10 | good | 0.0182 |
| 11 | won | 0.0181 |
| 12 | 500 | 0.0174 |
| 13 | 000 | 0.0163 |
| 14 | just | 0.0150 |
| 15 | exist | 0.0136 |

## Local SHAP Examples (Top 3 Messages)

Each example lists the strongest feature contributions for one message with respect to the **smishing** class.

### Example 1

- True label: `smishing`
- Predicted label: `smishing`
- smishing probability: 0.9953
- Clean text: _urgent your mobile number has embody awarded a ukp 2000 prize guaranteed call phonetoken from landline claim 3030 valid 12hrs only 150ppm_

| Feature | SHAP Value | Effect |
| --- | ---: | --- |
| PHONE | 1.1158 | Increases smishing score |
| phone_count | 0.7305 | Increases smishing score |
| claim | 0.3973 | Increases smishing score |
| prize | 0.3808 | Increases smishing score |
| URL | -0.3786 | Decreases smishing score |
| urgent | 0.2996 | Increases smishing score |
| landline | 0.2348 | Increases smishing score |
| awarded | 0.2155 | Increases smishing score |
| embody | 0.1913 | Increases smishing score |
| mobile number | 0.1825 | Increases smishing score |

### Example 2

- True label: `smishing`
- Predicted label: `smishing`
- smishing probability: 0.9946
- Clean text: _we ingest identified some unusual activity on your online banking log in via the secure link http phonetoken 81 urltoken to avoid account suspension_

| Feature | SHAP Value | Effect |
| --- | ---: | --- |
| URL | 2.1456 | Increases smishing score |
| PHONE | 1.1158 | Increases smishing score |
| url_count | -0.8037 | Decreases smishing score |
| phone_count | 0.7305 | Increases smishing score |
| link | 0.2949 | Increases smishing score |
| account | 0.2643 | Increases smishing score |
| ingest | 0.2576 | Increases smishing score |
| http | 0.2353 | Increases smishing score |
| log | 0.2273 | Increases smishing score |
| online | 0.1536 | Increases smishing score |

### Example 3

- True label: `smishing`
- Predicted label: `smishing`
- smishing probability: 0.9917
- Clean text: _urgent your mobile number has been grant with a inr 2 00 000 prize guaranteed call phonetoken from land line claim 3030 valid 12hrs only_

| Feature | SHAP Value | Effect |
| --- | ---: | --- |
| PHONE | 1.1158 | Increases smishing score |
| phone_count | 0.7305 | Increases smishing score |
| URL | -0.3786 | Decreases smishing score |
| claim | 0.3726 | Increases smishing score |
| prize | 0.3588 | Increases smishing score |
| urgent | 0.2830 | Increases smishing score |
| 000 | 0.2116 | Increases smishing score |
| mobile number | 0.1724 | Increases smishing score |
| mobile | 0.1676 | Increases smishing score |
| 000 prize | 0.1573 | Increases smishing score |

## Source Files

- `results\shap_outputs\global_mean_abs_shap.csv`
- `results\shap_outputs\top_smishing_features.csv`
- `results\shap_outputs\smishing_structured_feature_contributions.csv`
- `results\shap_outputs\local_smishing_explanations.csv`
