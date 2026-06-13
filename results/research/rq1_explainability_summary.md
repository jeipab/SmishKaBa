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
| PHONE | 0.8413 |
| phone_count | 0.6919 |
| URL | 0.6018 |
| url_count | 0.2535 |
| EMAIL | 0.0174 |
| email_count | 0.0174 |

**Interpretation:** Higher values indicate stronger average influence on smishing predictions. PHONE, URL, and related count features typically rank among the most influential structured indicators in smishing messages.

## Top 15 Influential TF-IDF Terms (Smishing Class)

| Rank | Term | Mean Abs SHAP |
| ---: | --- | ---: |
| 1 | you | 0.0819 |
| 2 | your | 0.0685 |
| 3 | claim | 0.0560 |
| 4 | to | 0.0521 |
| 5 | in | 0.0423 |
| 6 | and | 0.0368 |
| 7 | prize | 0.0299 |
| 8 | or | 0.0275 |
| 9 | the | 0.0267 |
| 10 | ingest | 0.0260 |
| 11 | have | 0.0259 |
| 12 | please | 0.0234 |
| 13 | my | 0.0202 |
| 14 | our | 0.0201 |
| 15 | message | 0.0201 |

## Local SHAP Examples (Top 3 Messages)

Each example lists the strongest feature contributions for one message with respect to the **smishing** class.

### Example 1

- True label: `smishing`
- Predicted label: `smishing`
- smishing probability: 0.9948
- Clean text: _you embody a 1000 winner or guaranteed caller prize this is our final attempt to contact you to claim call phonetoken now 150ppmpobox10183bhamb64xe_

| Feature | SHAP Value | Effect |
| --- | ---: | --- |
| PHONE | 1.1194 | Increases smishing score |
| phone_count | 0.7079 | Increases smishing score |
| URL | -0.3619 | Decreases smishing score |
| claim | 0.3318 | Increases smishing score |
| prize | 0.3079 | Increases smishing score |
| to claim | 0.2425 | Increases smishing score |
| phonetoken now | 0.2248 | Increases smishing score |
| you | 0.2084 | Increases smishing score |
| winner | 0.2051 | Increases smishing score |
| 1000 | 0.1742 | Increases smishing score |

### Example 2

- True label: `smishing`
- Predicted label: `smishing`
- smishing probability: 0.9948
- Clean text: _we have recalculated your vehicle tax you are owed 48 84 due to over urltoken the secure link http phonetoken to claim your refund_

| Feature | SHAP Value | Effect |
| --- | ---: | --- |
| URL | 2.0509 | Increases smishing score |
| PHONE | 1.1194 | Increases smishing score |
| url_count | -0.7536 | Decreases smishing score |
| phone_count | 0.7079 | Increases smishing score |
| claim | 0.3039 | Increases smishing score |
| link | 0.2476 | Increases smishing score |
| your | 0.2242 | Increases smishing score |
| to claim | 0.2233 | Increases smishing score |
| due | 0.2002 | Increases smishing score |
| http | 0.1980 | Increases smishing score |

### Example 3

- True label: `smishing`
- Predicted label: `smishing`
- smishing probability: 0.9935
- Clean text: _winner as a valued network customer you have been selected to receivea 900 prize reward to claim call phonetoken claim code kl341 valid 12 hours only_

| Feature | SHAP Value | Effect |
| --- | ---: | --- |
| PHONE | 1.1194 | Increases smishing score |
| phone_count | 0.7079 | Increases smishing score |
| claim | 0.4819 | Increases smishing score |
| URL | -0.3619 | Decreases smishing score |
| prize | 0.2525 | Increases smishing score |
| to claim | 0.1982 | Increases smishing score |
| winner | 0.1692 | Increases smishing score |
| you have | 0.1596 | Increases smishing score |
| been | 0.1497 | Increases smishing score |
| url_count | 0.1435 | Increases smishing score |

## Source Files

- `results\shap_outputs\global_mean_abs_shap.csv`
- `results\shap_outputs\top_smishing_features.csv`
- `results\shap_outputs\smishing_structured_feature_contributions.csv`
- `results\shap_outputs\local_smishing_explanations.csv`
