# Knowledge-enhanced feature engineering, agricultural TabPFN transfer, and evidence-bounded advisory for cross-regional maize yield forecasting

> Draft status: written strictly from `论文完整大纲.md` and the completed project artifacts. Items marked **TODO** require author-supplied metadata, references, or training records and have not been inferred.

# 1. Introduction

Subnational crop-yield forecasting supports food-security assessment, production planning, and climate-risk management. The increasing availability of meteorological, remote-sensing, soil, hydrological, crop-calendar, and yield-statistics data has enabled machine-learning models to represent interactions between seasonal growing conditions and crop production at administrative-unit scale [CITATION]. However, models developed for one region often lose accuracy when they are transferred to another climate or production environment [CITATION]. This limitation is especially important for international agricultural forecasting, where the amount of labelled data, the length of the yield record, the spatial units, and the dominant climate stresses differ among countries.

Cross-regional performance depends not only on the prediction algorithm but also on how agronomic features are constructed. Growing degree days, heat and cold stress indicators, precipitation stress, vegetation indices, and soil-moisture summaries commonly depend on fixed thresholds and aggregation rules [CITATION]. A single set of thresholds may be convenient for benchmarking, but it cannot be assumed to represent maize development and stress equally well in China, South Africa, Mexico, and Portugal. Region-adaptive feature engineering therefore provides a separate route to improving generalization: the predictive model may remain unchanged while the representation of the local growing environment is modified.

Large language models (LLMs) and retrieval-augmented generation (RAG) provide a mechanism for translating external agricultural knowledge and regional evidence into explicit feature-engineering configurations [CITATION]. In this study, the LLM did not predict yield. Instead, a structured retrieval procedure supplied the LLM with country-specific historical climate evidence, and the LLM generated candidate agronomic configurations within predefined numerical and categorical constraints. Candidate configurations specified temperature thresholds, precipitation-stress thresholds, and aggregation choices for vegetation and soil-moisture variables. Their value was determined only through downstream yield-prediction performance under forward temporal evaluation.

Feature-level adaptation does not address all sources of cross-regional variation. The forecasting model itself may also benefit from agricultural domain adaptation. Tabular foundation models, represented here by TabPFN, provide a pretrained alternative to conventional task-specific regressors [CITATION]. We continued the adaptation of TabPFN using Chinese maize data and refer to the resulting checkpoint as **AGRO-TabPFN**. With the feature representation fixed, the original and adapted checkpoints were evaluated in China and transferred without target-country adaptation to South Africa, Mexico, and Portugal.

Higher prediction accuracy alone does not constitute agricultural decision support. A yield forecast can identify risk, but it does not automatically identify which observed environmental conditions are associated with that risk. Similarly, unrestricted LLM text generation does not guarantee that a management suggestion is supported by the available data or is operationally feasible. A decision-support layer therefore requires an explicit boundary between prediction, diagnosis, conditional scenarios, and recommendation.

This study developed a three-stage framework on the basis of CY-Bench. Stage 1 used structured RAG and an LLM to generate region-adaptive feature configurations. Stage 2 compared the cross-regional performance of the original TabPFN and AGRO-TabPFN. Stage 3 converted forecasts into historical evidence-based risk diagnoses, bounded water-related scenarios, and grounded agricultural recommendations. The study addressed three questions: (1) can LLM–RAG-driven regional feature engineering improve crop-yield prediction across countries; (2) can a TabPFN checkpoint adapted with Chinese agricultural data transfer to South Africa, Mexico, and Portugal; and (3) can yield forecasts be converted into risk diagnoses and adaptation suggestions with explicit historical evidence and variable-operation boundaries?

# 2. Materials and methods

## 2.1 Resources search and database construction

The analysis used the maize component of CY-Bench for China (CN), South Africa (ZA), Mexico (MX), and Portugal (PT). The unit of analysis was an administrative region–year observation. The final country datasets contained 602 observations from 31 administrative units in CN (2003–2022), 167 observations from nine administrative units in ZA (2004–2022), 99 observations from 32 administrative units in MX (2014–2022; four observed years), and 88 observations from five administrative units in PT (2003–2020).

The predictor set combined meteorological, remote-sensing, soil, hydrological, and crop-calendar information. Meteorological variables included minimum, maximum, and mean temperature, precipitation, climatic water balance, and solar radiation. Remote-sensing variables included the normalized difference vegetation index (NDVI) and fraction of absorbed photosynthetically active radiation (FPAR). Soil and hydrological variables included surface soil moisture, available water capacity, and bulk density. Country-specific crop calendars defined the start and end of the maize season.

The source data were spatially matched to the CY-Bench administrative units and temporally aligned to each maize growing season. Seasonal predictors were divided into seven within-season periods. Static soil attributes were retained as fixed covariates, whereas meteorological, vegetation, and moisture variables were aggregated within the seasonal periods. The formal pipeline did not impute missing values. Records required by a downstream stage were checked for missing or non-finite values, and the stage stopped if a required field was incomplete. Each stage wrote CSV artifacts and adjacent manifests containing row and column identities, metadata, and SHA-256 hashes.

**TODO:** Insert the authoritative product name, provider, native spatial resolution, native temporal resolution, units, preprocessing citation, and access date for every CY-Bench input source.

**Table 1. Country-level structure of the maize yield database used in this study**

| Country | Administrative units | Region–year observations | Year range | Final RAG feature count |
|---|---:|---:|---:|---:|
| China (CN) | 31 | 602 | 2003–2022 | 100 |
| South Africa (ZA) | 9 | 167 | 2004–2022 | 112 |
| Mexico (MX) | 32 | 99 | 2014–2022 | 126 |
| Portugal (PT) | 5 | 88 | 2003–2020 | 126 |

## 2.2 Overall framework

The framework linked three forms of analysis. First, multi-source agricultural data were converted into country-specific feature tables. A structured RAG module retrieved historical climate evidence for the target country, after which the LLM proposed bounded feature-engineering configurations. Ridge regression, support vector regression (SVR), XGBoost, a one-dimensional convolutional neural network (CNN1D), and a flattened Transformer (TransformerFlat) evaluated the resulting representations. Second, the selected RAG feature table was held fixed while the original TabPFN and AGRO-TabPFN checkpoints were compared. Third, the prediction outputs were combined with strictly historical observations to diagnose yield risk, construct bounded water-related scenarios, and generate recommendations from a predefined action vocabulary.

**Figure 1. Overall framework of the proposed agricultural intelligence approach.**

```text
Multi-source agricultural data
        ↓
Regional information + structured RAG evidence
        ↓
LLM-generated, bounded feature configuration
        ↓
Ridge / SVR / XGBoost / CNN1D / TransformerFlat
        ↓
Fixed RAG feature representation
        ↓
TabPFN versus AGRO-TabPFN
        ↓
Historical reference and stress diagnosis
        ↓
Bounded scenario and evidence-grounded recommendation
```

## 2.3 RAG-enhanced LLM for adaptive feature engineering

### RAG knowledge base

The RAG evidence base was stored as a structured climate-metrics table (`climate_rag_metrics.csv`). It contained country- and crop-specific historical summaries relevant to maize temperature response, growing degree days, heat and cold conditions, precipitation and drought conditions, soil moisture, NDVI, and FPAR. The implemented retrieval procedure did not create embeddings and did not use a vector database. Instead, it filtered records by crop, country, and `year <= calibration_end_year`. Transparent scalar rules then selected up to six records representing typical, hot–dry, cold–wet, and prolonged-dry historical conditions. The retrieved rows, query constraints, source-table SHA-256, and complete evidence supplied to the LLM were saved for audit.

### Regional context construction

For each country, the outer test years were the final 30% of the available years and were never used for candidate generation or selection. The earlier calibration years were used to summarize temperature, precipitation, water balance, and stress–yield-residual relationships. These summaries formed the regional context supplied to the LLM. Thus, both the regional profile and retrieved evidence obeyed the same temporal boundary.

### RAG-enhanced feature-policy generation

Let (S_c) denote the calibration-period statistical context for country (c), (K_c) the structured records retrieved under the crop, country, and time filters, and (P_c) the candidate feature policy. The generation step was represented as

\[
P_c=f_{\mathrm{LLM}}(S_c,K_c).
\]

The same local Qwen3.5-9B base model was used for the LLM-only and RAG branches. The two branches differed only in whether the structured retrieval evidence was supplied. The LLM proposed three candidate configurations per non-fixed branch. Each candidate specified the maize base temperature and upper limit for growing degree days; minimum-temperature, maximum-temperature, and precipitation-stress rules; inclusion of FPAR, NDVI, and surface soil moisture; and aggregation methods for vegetation and soil-moisture variables. The program validated and bounded every generated field before feature construction. The LLM did not receive the outer-test results and did not directly estimate yield.

Candidate configurations were assessed within the calibration period. Each of the five downstream models independently selected the candidate with the lowest mean pooled calibration mean squared error over five repeats. A generic Ridge-selected RAG configuration was additionally saved as the fixed feature representation used by Stages 2 and 3. The selected generic policies are reported in Table 2; model-specific candidate identities were retained in the Stage 1 artifacts.

**Table 2. Generic RAG feature policies selected by Ridge regression and supplied to Stages 2 and 3**

| Country | GDD base (°C) | GDD upper limit (°C) | Cold rule (°C) | Heat rule (°C) | Precipitation rule | NDVI/FPAR aggregation | Soil-moisture aggregation |
|---|---:|---:|---|---|---|---|---|
| CN | 5 | 30 | `tmin < -4.027` | `tmax > 31.246` | `prec < 0.037` | median | mean |
| ZA | 10 | 35 | `tmin < 0` | `tmax > 35` | `prec < 1` | maximum | mean |
| MX | 10 | 30 | `tmin < -2.5` | `tmax > 35` | `prec < 0` | mean | mean |
| PT | 5 | 30 | `tmin < 0` | `tmax > 35` | `prec < 1` | mean | mean |

Three feature strategies were evaluated: fixed hardcoded features, Base-Qwen features generated without retrieval, and Qwen–RAG features. All strategies used the same five downstream model classes. Ridge used standardized predictors and (\alpha=1.0). SVR used standardized predictors, a radial-basis-function kernel, (C=10.0), (\epsilon=0.1), and `gamma="scale"`. XGBoost used 500 trees, a maximum depth of 4, and a learning rate of 0.03. CNN1D and TransformerFlat were trained for at most 50 epochs with internal early stopping. **TODO:** Insert the complete neural-network architecture, optimizer, batch size, loss, early-stopping patience, and remaining XGBoost settings from the finalized implementation record.

For every country and feature strategy, the last 30% of years formed the outer test set, satisfying (Year_{train}<Year_{test}). Five repeats used seeds 42, 1051, 2060, 3069, and 4078. The same temporal split and repeat seeds were applied across feature strategies within each model class.

## 2.4 AGRO-TabPFN construction and cross-regional transfer

Stage 2 compared the official TabPFN regression checkpoint with a checkpoint continued on Chinese agricultural data, denoted AGRO-TabPFN. Both checkpoints received the same country-specific generic RAG feature table saved by Stage 1. Therefore, the comparison changed the model checkpoint but not the feature construction within a country:

\[
X_{fixed,c}\rightarrow TabPFN
\]

and

\[
X_{fixed,c}\rightarrow AGRO\text{-}TabPFN.
\]

The Chinese adaptation dataset contained 482 training observations from 2003–2018 and 120 validation observations from 2019–2022. The evaluation stage verified that the original and adapted checkpoint hashes differed. The base checkpoint SHA-256 began `311ce18d…`, and the AGRO-TabPFN checkpoint SHA-256 began `53119ca4…`; complete hashes were retained in the artifact manifests. **TODO:** Supply the missing continuation-training record, including the TabPFN implementation/version, objective, optimizer, learning rate, batch size, epoch or step count, early-stopping rule, hardware, base-checkpoint identity, and checkpoint-selection procedure. The current `cn_finetuning_manifest.json` contains null values for these trainer fields.

China was reported as the source-domain reference. Cross-regional evaluation was conducted in ZA, MX, and PT without further target-country adaptation. The final 30% of each country's years formed the temporal test set. Five repeats used the same seed schedule as Stage 1. Transfer gain for country (c) was calculated as

\[
TG_c=R^2_{AGRO,c}-R^2_{TabPFN,c},
\]

and MAPE improvement was calculated as

\[
\Delta MAPE_c=MAPE_{TabPFN,c}-MAPE_{AGRO,c}.
\]

Positive values of both quantities indicated consistent positive transfer. Opposite signs indicated metric-dependent transfer.

## 2.5 Historical evidence-based yield-risk diagnosis

For each target record ((adm_i,t)), Stage 3 used only observations with (year<t). A local historical reference was used when at least five earlier observations were available for the same administrative unit; otherwise, the reference pool was expanded to earlier observations from the same country. High-yield reference records were selected relative to the 0.75 yield quantile by default, with LLM-proposed values clipped to the interval 0.65–0.90. Feature-reference targets were represented by the median of the historical high-yield pool, and dispersion was estimated from the interquartile range or standard deviation.

An ExtraTrees yield-response surrogate (\hat y=f(X)) was fitted using 300 trees, all available processor cores, and the repeat-specific seed. The most recent 30% of training-period years were used for forward validation. The final surrogate was fitted only to records strictly earlier than the earliest decision record; target-year observed yield did not enter model fitting.

For feature (j), the stress score combined the standardized deviation from the high-yield reference ((D_j)), ExtraTrees feature importance ((I_j)), and the positive change in surrogate prediction when that feature alone was moved toward its reference ((G_j)):

\[
S_j=D_j\times I_j\times G_j.
\]

The resulting scores ranked the observed water-related, temperature, radiation, soil, and crop-state deviations for each decision record.

The scenario layer allowed numerical changes only to water-related proxies, defined by feature names containing precipitation, climatic water balance, or surface soil moisture. Temperature, radiation, static soil properties, NDVI, and FPAR were held at their observed values. Each water proxy could move no more than 50% of the distance toward its historical reference by default, and the candidate value was clipped to the 10th–90th percentile of the earlier observed distribution. A change was accepted only if (\hat y(X')>\hat y(X)). These outputs represent predicted conditional responses under a bounded model scenario; they do not represent irrigation doses or causal yield effects.

## 2.6 Evidence-grounded agricultural advisory

The advisory module supplied the LLM with structured evidence fields including the administrative-unit identifier, year, diagnosed feature, stress direction, and accepted scenario. The LLM could select only from the predefined actions: irrigation-feasibility assessment, soil-moisture conservation, drainage assessment, sowing-date review, stress-tolerant cultivar review, and field monitoring. A generated item was grounded only when its administrative unit and year matched the decision record, its feature was present in the diagnostic evidence, and its action belonged to the allowed vocabulary. Outputs failing any rule were classified as unsupported recommendations.

Base Qwen and Qwen–RAG used the same Qwen3.5-9B checkpoint. The RAG branch additionally received country-, crop-, and time-filtered rows from the structured climate table. Grounding was evaluated independently of the agronomic effectiveness of an action. Thus, a grounded recommendation was traceable to available evidence, but it was not treated as a validated causal intervention.

## 2.7 Evaluation metrics and statistical analysis

Yield-prediction performance was evaluated using the coefficient of determination and mean absolute percentage error:

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2},
\]

\[
MAPE=\frac{100}{n}\sum_i\left|\frac{y_i-\hat y_i}{y_i}\right|.
\]

The tables report the mean across five repeats; the project artifacts also retain the sample variance (`ddof=1`) for every metric. Stage 3 additionally reported the number of decision records, mean predicted conditional response, number of stress signals, schema-validity rate, grounded-recommendation rate, and unsupported-recommendation rate. **TODO:** If inferential claims are required, add the prespecified paired country–model comparison or confidence-interval procedure. The completed pipeline currently reports repeated-run means and sample variances but no hypothesis test.

# 3. Results

## 3.1 Effects of RAG-enhanced adaptive feature engineering on yield prediction

The effect of RAG-enhanced feature engineering depended on both country and downstream model (Table 3). In CN, Qwen–RAG produced the highest (R^2) for CNN1D (0.524), SVR (0.640), and TransformerFlat (0.550), while the hardcoded strategy remained slightly better for Ridge and XGBoost. The clearest CN gain occurred for SVR, where (R^2) increased from 0.601 with hardcoded features to 0.640 with Qwen–RAG and MAPE decreased from 9.49% to 8.95%.

In ZA, the no-retrieval Base-Qwen configuration was strongest for Ridge, SVR, and XGBoost. Qwen–RAG improved SVR over the hardcoded baseline ((R^2=0.619) versus 0.528; MAPE 16.69% versus 18.16%) but did not improve CNN1D, Ridge, or TransformerFlat. In MX, Qwen–RAG gave the highest (R^2) and lowest MAPE for SVR and the highest (R^2) for XGBoost, but its CNN1D and TransformerFlat results were weaker than the corresponding hardcoded results. In PT, Qwen–RAG improved Ridge and XGBoost over the hardcoded strategy, whereas CNN1D and SVR deteriorated and all TransformerFlat configurations produced negative (R^2).

Across the 20 country–model combinations, Qwen–RAG exceeded the hardcoded strategy in (R^2) for 10 combinations and achieved a lower MAPE for 9 combinations. These results show that RAG-guided feature engineering was not uniformly beneficial; its effects were conditional on the country and predictor.

**Table 3. Test performance under hardcoded, Base-Qwen, and Qwen–RAG feature-engineering strategies**

| Country | Model | Hardcoded R² | Hardcoded MAPE (%) | Base Qwen R² | Base Qwen MAPE (%) | Qwen–RAG R² | Qwen–RAG MAPE (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| CN | CNN1D | 0.484 | 11.05 | 0.477 | 11.20 | 0.524 | 10.63 |
| CN | Ridge | 0.484 | 11.03 | 0.452 | 11.31 | 0.453 | 11.36 |
| CN | SVR | 0.601 | 9.49 | 0.605 | 9.37 | 0.640 | 8.95 |
| CN | TransformerFlat | 0.510 | 10.18 | 0.506 | 10.15 | 0.550 | 10.00 |
| CN | XGBoost | 0.618 | 9.21 | 0.593 | 9.43 | 0.598 | 9.38 |
| ZA | CNN1D | 0.515 | 18.16 | 0.455 | 19.21 | 0.431 | 19.26 |
| ZA | Ridge | 0.595 | 17.82 | 0.619 | 16.64 | 0.595 | 17.82 |
| ZA | SVR | 0.528 | 18.16 | 0.625 | 16.31 | 0.619 | 16.69 |
| ZA | TransformerFlat | 0.509 | 18.77 | 0.475 | 19.64 | 0.509 | 18.77 |
| ZA | XGBoost | 0.410 | 17.65 | 0.589 | 17.00 | 0.457 | 17.05 |
| MX | CNN1D | 0.482 | 68.09 | 0.487 | 67.12 | 0.469 | 71.58 |
| MX | Ridge | 0.708 | 47.56 | 0.698 | 56.82 | 0.663 | 50.17 |
| MX | SVR | 0.719 | 47.64 | 0.729 | 51.01 | 0.732 | 45.34 |
| MX | TransformerFlat | 0.036 | 83.05 | 0.094 | 79.09 | 0.088 | 83.13 |
| MX | XGBoost | 0.611 | 60.18 | 0.555 | 57.69 | 0.643 | 57.68 |
| PT | CNN1D | 0.212 | 25.44 | 0.306 | 25.07 | 0.183 | 26.33 |
| PT | Ridge | 0.360 | 24.52 | 0.379 | 22.42 | 0.401 | 22.35 |
| PT | SVR | 0.384 | 20.89 | 0.364 | 20.88 | 0.346 | 21.27 |
| PT | TransformerFlat | -0.013 | 29.63 | -0.043 | 29.41 | -0.033 | 29.74 |
| PT | XGBoost | 0.316 | 22.10 | 0.337 | 21.25 | 0.408 | 20.57 |

**Figure 2. Country-specific changes in (R^2) and MAPE produced by Base-Qwen and Qwen–RAG feature engineering relative to the hardcoded strategy.** The existing project figure can be inserted from `figures/fig_stage1_r2_mape_cn_za_mx_pt.*`.

## 3.2 Regional characteristics of RAG-guided feature configurations

The RAG-guided configurations differed at two levels: the candidate policies generated for each country and the candidate selected by each downstream model. Three recurring policy types were produced. The local-profile policy primarily used quantiles and event frequencies from the country-specific calibration period; the historical-regime policy placed greater weight on the retrieved cool–wet, hot–dry, and long-dry-spell years; and the robust-hybrid policy combined local thresholds with longer-term regime statistics. Consequently, regional adaptation was not represented by a single country-level configuration. Table 2 reports only the Ridge-selected generic configuration subsequently supplied to Stages 2 and 3, whereas the formal Stage 1 comparison used model-specific selections (Supplementary Table S2).

**Supplementary Table S2. Model-specific RAG candidate selected using calibration-period prediction error**

| Country | Ridge | XGBoost | SVR | CNN1D | TransformerFlat |
|---|---|---|---|---|---|
| CN | Local profile | Robust hybrid | Historical regime | Historical regime | Historical regime |
| ZA | Local profile | Robust hybrid | Historical regime | Robust hybrid | Local profile |
| MX | Historical regime | Historical regime | Local profile | Local profile | Historical regime |
| PT | Local profile | Historical regime | Local profile | Historical regime | Local profile |

The CN candidates reflected the wide thermal range and strong precipitation variability present in the calibration evidence. Across 422 calibration-period location–year observations, the fifth percentile of minimum temperature was −7.80 °C, the 95th percentile of maximum temperature was 31.25 °C, and the lower quartile of daily precipitation was 0.037 mm. The retrieved records also separated cool–wet, hot–dry, and prolonged-dry conditions. Accordingly, the local-profile policy reduced the hardcoded maize GDD range from 10–35 °C to 5–30 °C and used thresholds of −4.027 °C, 31.246 °C, and 0.037 mm for cold, heat, and low-precipitation events, respectively. It also replaced maximum NDVI and FPAR aggregation with the median. The historical-regime policy used a wider 6.32–35 °C GDD interval, less extreme temperature thresholds, a substantially higher precipitation threshold of 6.425 mm, and median aggregation for both vegetation and soil moisture. This historical-regime candidate was selected by SVR, CNN1D, and TransformerFlat, for which R² increased by 0.039, 0.040, and 0.040 relative to the hardcoded features. In contrast, the local-profile policy selected by Ridge reduced R² by 0.030, and the robust-hybrid policy selected by XGBoost reduced R² by 0.020. Thus, the CN evidence supported several internally distinct feature hypotheses, but their predictive value depended on the downstream model.

ZA showed a different pattern. The retrieved history contained distinct cool–wet and hot–dry years, including a hot–dry record with 30 days above 35 °C, while the calibration data showed negative yield-residual correlations for heat and low-precipitation event counts. For example, the `tmax > 31.58 °C` event count had a residual correlation of −0.219, and precipitation thresholds between 0.37 and 2.00 mm had correlations ranging from −0.345 to −0.369. The robust-hybrid candidate incorporated these empirical limits through a 12–40 °C GDD range, a heat threshold of 31.58 °C, and a precipitation threshold of 0.37 mm. Nevertheless, the Ridge-selected local-profile policy exactly reproduced the hardcoded settings: a 10–35 °C GDD range, 0 °C and 35 °C thermal-stress thresholds, a 1-mm precipitation threshold, maximum vegetation aggregation, and mean soil-moisture aggregation. Ridge and TransformerFlat selected this unchanged policy and consequently reproduced the hardcoded results exactly. The calibration procedure therefore retained the baseline when the alternatives did not improve validation performance instead of forcing a nominally regional modification. By comparison, the historical-regime policy increased SVR R² from 0.528 to 0.619, whereas the same broader family of regional alternatives was not uniformly beneficial: the robust-hybrid policy increased XGBoost R² by 0.047 but reduced CNN1D R² by 0.085.

The MX candidates were differentiated more by water-stress representation and aggregation choice than by thermal accumulation. The calibration profile had a median climatic water balance of −4.479 mm and approximately 106 crop-season days with precipitation below 1 mm, whereas retrieved hot–dry and long-dry-spell years captured substantial interannual moisture variability. All RAG candidates used a GDD base of 10 °C and an upper limit of 30 °C. The local-profile policy retained a 1-mm low-precipitation threshold and used maximum aggregation for vegetation and soil moisture, while the historical-regime policy used mean aggregation and set the precipitation threshold to `prec < 0`. Because the precipitation input was non-negative, this rule did not generate low-precipitation events and effectively removed that particular stress indicator from the historical-regime representation. The local-profile policy was selected by both SVR and CNN1D, yet their responses diverged: SVR R² increased by 0.012 and MAPE decreased by 2.30 percentage points, whereas CNN1D R² decreased by 0.013 and MAPE increased by 3.67 percentage points. The historical-regime policy similarly increased XGBoost R² by 0.032 but reduced Ridge R² by 0.046. TransformerFlat gained 0.052 in R² under this policy, but its MAPE did not improve and remained above 83%. These opposing responses show that a configuration cannot be considered effective solely because it is region specific.

PT exhibited a further separation between a conservative local profile and a wider historical-regime policy. Its calibration evidence described rare frost events, recurrent low-precipitation days, and occasional summer heat extremes. The local-profile policy therefore used a 5–30 °C GDD range, retained the conventional 0 °C, 35 °C, and 1-mm stress thresholds, and replaced maximum vegetation aggregation with the mean. The historical-regime candidate instead used a 10–50 °C GDD range, thresholds of −2 °C, 33 °C, and 2 mm, median vegetation aggregation, and maximum soil-moisture aggregation. The local-profile policy increased Ridge R² by 0.041 and reduced MAPE by 2.17 percentage points, but it reduced the performance of SVR and TransformerFlat. Conversely, the historical-regime policy increased XGBoost R² by 0.092 and reduced MAPE by 1.53 percentage points, while reducing CNN1D R² by 0.029. The same country-level climate evidence therefore supported alternative parameterizations with opposite effects across model classes.

Taken together, these results show that RAG changed more than the numerical values of agronomic thresholds. It altered which portions of the regional climate distribution were emphasized, whether seasonal variables were represented by mean, median, or maximum aggregation, and in the MX historical-regime candidate, whether a precipitation-stress event feature remained active. However, the resulting performance changes were conditional on both country and downstream model. The evidence therefore supports interpreting the generated configurations as traceable, testable feature-engineering hypotheses rather than as universally optimal agronomic parameter sets or recovered crop-physiological mechanisms.

## 3.3 Cross-regional performance of AGRO-TabPFN

AGRO-TabPFN improved both R² and MAPE in CN, ZA, MX, and PT (Supplementary Table S1). In the CN source domain, R² increased from 0.874 to 0.885 and MAPE decreased from 5.70% to 5.49%. The largest target-country increase in R² occurred in ZA, from 0.668 to 0.684, followed by PT, from 0.454 to 0.467. MX showed a smaller increase from 0.860 to 0.863.

**Supplementary Table S1. Cross-regional test performance of TabPFN and AGRO-TabPFN**

| Country | Test years | TabPFN R² | AGRO-TabPFN R² | ΔR² | TabPFN MAPE (%) | AGRO-TabPFN MAPE (%) | MAPE improvement (percentage points) |
|---|---|---:|---:|---:|---:|---:|---:|
| CN | 2020–2022 | 0.874 | 0.885 | 0.011 | 5.70 | 5.49 | 0.21 |
| ZA | 2017–2022 | 0.668 | 0.684 | 0.016 | 16.20 | 15.65 | 0.55 |
| MX | 2022 | 0.860 | 0.863 | 0.003 | 31.75 | 31.40 | 0.35 |
| PT | 2016–2020 | 0.454 | 0.467 | 0.013 | 20.52 | 20.16 | 0.36 |

**Figure 3. Cross-regional prediction performance of TabPFN and AGRO-TabPFN in CN, ZA, MX, and PT.**

## 3.4 Transfer characteristics of AGRO-TabPFN

The direction of transfer was positive under both metrics in all three target countries included in the outline, but its magnitude varied. ZA had the largest joint gain ((\Delta R^2=0.0156); MAPE improvement 0.546 percentage points). PT had a similar (R^2) gain ((\Delta R^2=0.0132)) but a smaller MAPE improvement of 0.358 percentage points. MX showed only a small gain ((\Delta R^2=0.0032); MAPE improvement 0.350 percentage points). The results therefore support cross-regional transfer from the CN-adapted checkpoint, but not a claim of equal transfer strength across target regions.

**TODO:** The outline proposes relating transfer gain to temperature regime, rainfall regime, crop calendar, and climate similarity. Insert this analysis only if a completed artifact containing the corresponding similarity measures is supplied.

## 3.5 Yield-risk diagnosis based on historical evidence

The ExtraTrees surrogate retained useful forward predictive ability in all four countries, although performance differed substantially. Mean forward (R^2) was 0.765 in CN, 0.721 in MX, 0.591 in ZA, and 0.415 in PT. The corresponding RMSE values were 0.622, 1.644, 1.367, and 2.764 yield units, respectively. These results established the surrogate as a conditional response model for the diagnostic layer, while the lower performance in PT required greater caution in interpreting local scenario magnitudes.

The diagnostic stage generated 926 Qwen–RAG stress signals for 31 CN decision records, 1,441 signals for 32 MX records, 336 signals for nine ZA records, and 155 signals for five PT records. Signals covered water-related conditions, temperature, radiation, static soil attributes, and crop-state indicators. Only water-related proxy signals were eligible for numerical scenario changes; the remaining categories informed non-numeric adaptation actions.

**Figure 4. Dominant yield-risk factors across CN, ZA, MX, and PT.** **TODO:** Generate the final country-level stress-category aggregation from `stress_diagnostics.csv` and insert the resulting panel.

## 3.6 Predicted responses under constrained agricultural scenarios

The bounded water-recovery scenarios produced positive mean conditional responses in each country (Table 4). Under the Qwen–RAG policy, the mean response was 0.021 yield units in CN, 0.368 in MX, 0.062 in PT, and 0.087 in ZA. The corresponding post-scenario mean predictions were 5.231, 4.581, 9.392, and 6.420 yield units. These values quantify the response of the fitted surrogate when accepted water-related proxies were partially moved toward historically observed high-yield conditions. They are not causal yield improvements and do not specify irrigation quantities.

The relationship between policies also varied by country. Qwen–RAG and the deterministic policy produced the same aggregate scenario response in MX, PT, and ZA. In CN, the deterministic policy produced a larger mean response (0.030) than Qwen–RAG (0.021), whereas Base Qwen produced 0.019. Thus, the RAG branch did not uniformly increase scenario magnitude; its purpose was to constrain the evidence-to-policy path rather than maximize the numerical response.

**Table 4. Grounding reliability and bounded water-scenario results for generated agricultural advisory**

| Country | Policy | Decision records | Mean predicted conditional response | Stress signals | Recommendations | Grounded recommendations | Grounded rate |
|---|---|---:|---:|---:|---:|---:|---:|
| CN | Base Qwen | 31 | 0.019 | 926 | 6 | 6 | 100% |
| CN | Qwen–RAG | 31 | 0.021 | 926 | 6 | 6 | 100% |
| ZA | Base Qwen | 9 | 0.087 | 336 | 6 | 6 | 100% |
| ZA | Qwen–RAG | 9 | 0.087 | 336 | 6 | 6 | 100% |
| MX | Base Qwen | 32 | 0.111 | 1,488 | 6 | 6 | 100% |
| MX | Qwen–RAG | 32 | 0.368 | 1,441 | 6 | 6 | 100% |
| PT | Base Qwen | 5 | 0.015 | 182 | 5 | 5 | 100% |
| PT | Qwen–RAG | 5 | 0.062 | 155 | 5 | 5 | 100% |

## 3.7 Reliability of evidence-grounded agricultural advisory

All saved Base-Qwen and Qwen–RAG advisory outputs for the four countries passed the output-schema checks. Every generated recommendation matched the required administrative unit, year, diagnosed feature, and allowed action vocabulary, producing a grounded-recommendation rate of 100% and an unsupported-recommendation rate of 0% in the completed artifacts. The system therefore prevented free-form numerical interventions such as arbitrary temperature reduction and limited the output to bounded actions such as irrigation-feasibility assessment, soil-moisture conservation, drainage assessment, sowing-date review, stress-tolerant cultivar review, and field monitoring.

This result measures formal grounding reliability rather than agronomic effectiveness. The number of generated recommendations was small (five or six per branch and country), and the protocol did not include completed expert ratings. **TODO:** Insert the preregistered blinded agronomist evaluation after at least three independent reviewers have scored relevance, actionability, safety, evidence grounding, and overreach. Do not replace this evaluation with another LLM.

# 4. Discussion

## 4.1 Contribution of RAG-enhanced knowledge to adaptive feature engineering

The Stage 1 results show that structured RAG can change explicit agronomic feature policies, but that these changes do not guarantee a universal predictive gain. The RAG branch improved several country–model combinations, including CN SVR and TransformerFlat, ZA SVR, MX SVR and XGBoost, and PT Ridge and XGBoost. In other combinations, hardcoded or Base-Qwen features performed better. The main contribution of RAG in this framework is therefore not an unconditional increase in accuracy. It is a traceable mechanism for converting regional historical evidence into bounded feature-engineering alternatives whose value can be tested under a leakage-controlled temporal protocol.

Differences among downstream models indicate that feature-policy quality is conditional on the predictor. Linear Ridge regression, kernel SVR, boosted trees, CNN1D, and TransformerFlat respond differently to changes in thresholds, aggregation functions, and feature sets. This interaction explains why a configuration selected for one model should not be assumed optimal for another. The implementation addressed this issue in Stage 1 through model-specific candidate selection, while retaining a single Ridge-selected representation for the controlled Stage 2 checkpoint comparison.

The comparison between Base Qwen and Qwen–RAG further separates language-model generation from retrieval. In CN, Qwen–RAG exceeded Base Qwen in (R^2) for all five models, but this pattern did not extend to ZA, MX, or PT. RAG therefore supplied explicit evidence rather than retraining the LLM, and its empirical benefit depended on whether the retrieved country history supported a feature policy compatible with the downstream model and evaluation period.

## 4.2 Agricultural adaptation and cross-regional transfer of AGRO-TabPFN

Adapting TabPFN with Chinese agricultural data improved both reported metrics in ZA, MX, and PT while also improving the CN source-domain result. The largest target-domain gain occurred in ZA, and the smallest occurred in MX. These results support the existence of cross-regional transfer, but the unequal magnitudes define a transfer boundary: agricultural adaptation in one country does not imply uniform global improvement.

The fixed-feature design strengthens this interpretation because the original and adapted checkpoints received the same Stage 1 RAG representation within each country. The observed differences can therefore be attributed to checkpoint choice within the implemented evaluation rather than to a change in country-specific feature construction. However, the current training-provenance record is incomplete. The adaptation claim must remain provisional in the manuscript until the optimizer, schedule, checkpoint-selection process, and software-version metadata are restored.

## 4.3 Two complementary forms of agricultural knowledge incorporation

The framework evaluates two distinct routes for incorporating agricultural knowledge. The first route is explicit knowledge augmentation:

\[
External\ evidence\rightarrow structured\ RAG\rightarrow LLM\rightarrow feature\ engineering.
\]

Its output is inspectable: temperature thresholds, precipitation rules, inclusion decisions, and aggregation functions can be compared across countries and traced to retrieved records. Its limitation is that a plausible policy may interact unfavourably with a particular prediction model.

The second route is model-level domain adaptation:

\[
Chinese\ agricultural\ data\rightarrow TabPFN\ adaptation\rightarrow AGRO\text{-}TabPFN.
\]

This route changes the model checkpoint while holding the feature representation fixed. It was more consistent across the three target countries in this study, although the effect sizes were small and region dependent. The two routes are therefore complementary in location and interpretability: RAG changes the explicit input representation, whereas continued TabPFN adaptation changes model behaviour. Their results should not be collapsed into a single claim of “foundation-model improvement.”

## 4.4 From yield prediction to evidence-bounded agricultural advisory

The Stage 3 design separates prediction, diagnosis, and decision. The TabPFN outputs supplied forecast context. A separate leakage-controlled ExtraTrees surrogate linked observed features to a historical high-yield reference and ranked deviations. Only water-related proxies could be modified numerically, every proposed value remained within an earlier observed range, and a scenario was accepted only when the surrogate predicted a positive conditional response. Finally, the LLM selected actions from a fixed vocabulary, and each output was validated against its evidence fields.

This sequence prevents two forms of overreach. First, it does not treat a forecast error or low predicted yield as a direct explanation of crop stress. Second, it does not translate a model scenario into a causal management prescription. The reported scenario gains indicate how the surrogate responded when selected water proxies moved toward historical high-yield conditions. They do not show that a specific irrigation amount would cause the same yield change. Likewise, a 100% grounding rate shows that the recommendation text conformed to the stored evidence and action constraints; it does not show that the recommended action is agronomically effective in the field.

## 4.5 Limitations and future work

This study has six main limitations. First, the manuscript focuses on CN, ZA, MX, and PT, so the findings do not establish global generalization. Second, AGRO-TabPFN used only China as the adaptation source. Third, the scope and quality of the structured RAG table constrain the feature policies that the LLM can generate. Fourth, CY-Bench does not contain observed irrigation, fertilizer, cultivar, or field-management variables. Fifth, Stage 3 relies on observational data and a predictive surrogate and cannot identify causal intervention effects. Sixth, the advisory outputs have not yet been validated in field experiments.

Additional limitations arise from the completed evaluation. MX contains only four observed years and a single outer test year (2022), which limits temporal generalization claims despite the number of administrative units. PT contains only five decision records in Stage 3, and its surrogate achieved the lowest forward (R^2) among the four countries. The five computational repeats quantify seed-related variability but do not replace independent temporal or geographic replications. Finally, the 100% grounding rate was obtained on a small set of tightly constrained outputs and requires blinded expert evaluation for relevance, actionability, safety, and overreach.

Future work should evaluate additional source and target countries, multiple crops, and multi-source adaptation. Incorporating observed management variables would permit more direct analysis of irrigation, fertilizer, cultivar, and planting decisions. Process-based crop models could be combined with the data-driven framework to impose stronger physiological constraints. Expert review and prospective field experiments are required before the generated actions can be considered operational recommendations.

# 5. Conclusions

This study examined three linked uses of foundation models in cross-regional maize yield forecasting. First, structured RAG enabled Qwen3.5-9B to generate auditable, region-specific feature policies without using the LLM as a yield predictor. These policies improved some country–model combinations but not others, showing that knowledge-enhanced feature engineering is model and region dependent. Second, AGRO-TabPFN improved both (R^2) and MAPE relative to the original TabPFN in CN, ZA, MX, and PT, with unequal gains that define a cross-regional transfer boundary. Third, the prediction outputs were converted into historical-reference diagnoses, bounded water-related model scenarios, and recommendations that were validated against explicit evidence fields and an allowed action vocabulary.

The results support a broader evaluation principle for agricultural foundation models. Their value should not be judged only by accuracy in a single region. Evaluation should also determine whether external agricultural knowledge can be incorporated transparently, whether model adaptation transfers across distinct production environments, and whether predictions can be converted into decision information without crossing the boundary from statistical grounding to unsupported causal advice.
