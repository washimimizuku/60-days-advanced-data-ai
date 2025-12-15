# Day 27 Quiz: Time Series Forecasting - ARIMA, Prophet & Neural Networks

## Questions

### 1. What is the primary advantage of using SARIMA over ARIMA for time series forecasting?
- a) SARIMA is computationally faster than ARIMA
- b) SARIMA can handle seasonal patterns in addition to trend and noise
- c) SARIMA requires fewer parameters to tune
- d) SARIMA works better with non-stationary data

### 2. In Prophet forecasting, what is the purpose of adding custom seasonality components?
- a) To reduce the computational complexity of the model
- b) To capture domain-specific cyclical patterns not covered by built-in seasonalities
- c) To eliminate the need for holiday effects
- d) To make the model more interpretable for business users

### 3. When preparing sequences for LSTM forecasting, why is the sequence length (lookback window) important?
- a) Longer sequences always produce better forecasts
- b) The sequence length determines how much historical context the model can use to make predictions
- c) Shorter sequences reduce overfitting in all cases
- d) The sequence length must equal the forecast horizon

### 4. What is the main benefit of using ensemble forecasting methods?
- a) Ensemble methods are always faster than individual models
- b) Ensemble methods combine the strengths of different approaches and reduce individual model biases
- c) Ensemble methods require less data for training
- d) Ensemble methods eliminate the need for model validation

### 5. In ARIMA model selection, what does the AIC (Akaike Information Criterion) help determine?
- a) The optimal forecast horizon
- b) The best balance between model complexity and goodness of fit
- c) The required amount of training data
- d) The confidence intervals for forecasts

### 6. Why is stationarity important for classical time series forecasting methods like ARIMA?
- a) Stationary data is easier to visualize
- b) Non-stationary data requires more computational resources
- c) ARIMA assumes constant statistical properties over time, which stationary data provides
- d) Stationary data always has better forecast accuracy

### 7. What is the purpose of Monte Carlo dropout in LSTM forecasting?
- a) To speed up model training
- b) To estimate prediction uncertainty by generating multiple forecasts with different dropout patterns
- c) To reduce the model size for deployment
- d) To eliminate the need for validation data

### 8. In Prophet, what role do changepoints serve in the forecasting model?
- a) Changepoints define seasonal patterns
- b) Changepoints identify points where the trend rate changes significantly
- c) Changepoints determine holiday effects
- d) Changepoints set the forecast confidence intervals

### 9. When combining forecasts in an ensemble, what is the advantage of weighted averaging over simple averaging?
- a) Weighted averaging is computationally simpler
- b) Weighted averaging allows better-performing models to have more influence on the final forecast
- c) Weighted averaging eliminates the need for individual model validation
- d) Weighted averaging always produces more accurate forecasts

### 10. What is the most critical consideration when implementing time series forecasting in production?
- a) Using the most complex model available
- b) Ensuring consistent data preprocessing and handling of missing values across training and serving
- c) Maximizing the forecast horizon
- d) Using only neural network-based approaches

---

## Answers

### 1. What is the primary advantage of using SARIMA over ARIMA for time series forecasting?
**Answer: b) SARIMA can handle seasonal patterns in addition to trend and noise**

**Explanation:** SARIMA (Seasonal ARIMA) extends ARIMA by adding seasonal components that can model repeating patterns at fixed intervals (e.g., weekly, monthly, yearly seasonality). While ARIMA can handle trend and noise through its AR, I, and MA components, it cannot capture seasonal patterns. SARIMA adds seasonal AR, I, and MA terms that specifically model these cyclical behaviors. This makes SARIMA particularly valuable for business data that often exhibits strong seasonal patterns, such as retail sales, energy consumption, or website traffic. The seasonal components allow the model to learn that, for example, sales are typically higher in December or website traffic peaks on certain days of the week.

---

### 2. In Prophet forecasting, what is the purpose of adding custom seasonality components?
**Answer: b) To capture domain-specific cyclical patterns not covered by built-in seasonalities**

**Explanation:** Prophet comes with built-in seasonality for yearly, weekly, and daily patterns, but many business domains have unique cyclical patterns that don't fit these standard categories. Custom seasonality allows you to model domain-specific cycles such as bi-weekly payroll effects, quarterly business cycles, academic semester patterns, or industry-specific seasonal behaviors. For example, a retail business might have a 6-week promotional cycle, or a B2B company might see monthly billing cycles. By adding custom seasonality with appropriate period and Fourier order parameters, you can capture these patterns that would otherwise be treated as noise, leading to more accurate forecasts and better business insights.

---

### 3. When preparing sequences for LSTM forecasting, why is the sequence length (lookback window) important?
**Answer: b) The sequence length determines how much historical context the model can use to make predictions**

**Explanation:** The sequence length in LSTM forecasting defines how many previous time steps the model considers when making a prediction. This is crucial because it determines the temporal context available to the model. Too short a sequence might miss important long-term patterns and dependencies, while too long a sequence might include irrelevant historical information and increase computational complexity. The optimal sequence length depends on the data characteristics: for daily sales data, you might need 30-90 days to capture monthly patterns, while for hourly data, you might need 24-168 hours to capture daily and weekly patterns. The sequence length should be long enough to capture the relevant temporal dependencies but not so long that it includes noise or makes training inefficient.

---

### 4. What is the main benefit of using ensemble forecasting methods?
**Answer: b) Ensemble methods combine the strengths of different approaches and reduce individual model biases**

**Explanation:** Ensemble forecasting leverages the principle that different models capture different aspects of the data and make different types of errors. By combining multiple models (e.g., ARIMA for linear trends, Prophet for seasonality, LSTM for non-linear patterns), the ensemble can achieve better overall performance than any individual model. Each model's weaknesses are compensated by other models' strengths, and random errors tend to cancel out when averaged. This leads to more robust and reliable forecasts, especially in production environments where consistent performance across different data conditions is crucial. Ensemble methods also provide natural uncertainty quantification through the variance of individual model predictions.

---

### 5. In ARIMA model selection, what does the AIC (Akaike Information Criterion) help determine?
**Answer: b) The best balance between model complexity and goodness of fit**

**Explanation:** AIC is an information criterion that balances model fit quality against model complexity by penalizing models with more parameters. In ARIMA model selection, you typically test multiple combinations of (p,d,q) parameters, and AIC helps identify which combination provides the best trade-off. A model with more parameters might fit the training data better but could overfit and perform poorly on new data. AIC addresses this by adding a penalty term proportional to the number of parameters. Lower AIC values indicate better models. This automated selection process is crucial for production systems where you need objective, repeatable model selection criteria rather than manual parameter tuning.

---

### 6. Why is stationarity important for classical time series forecasting methods like ARIMA?
**Answer: c) ARIMA assumes constant statistical properties over time, which stationary data provides**

**Explanation:** ARIMA models are built on the assumption that the underlying statistical properties of the time series (mean, variance, autocorrelation structure) remain constant over time. Non-stationary data violates this assumption because its statistical properties change over time - for example, an increasing trend means the mean is not constant. When applied to non-stationary data, ARIMA models can produce unreliable forecasts and spurious relationships. This is why differencing (the 'I' in ARIMA) is used to transform non-stationary data into stationary data by removing trends. Stationarity ensures that patterns learned from historical data will remain valid for future predictions, making the forecasts more reliable and theoretically sound.

---

### 7. What is the purpose of Monte Carlo dropout in LSTM forecasting?
**Answer: b) To estimate prediction uncertainty by generating multiple forecasts with different dropout patterns**

**Explanation:** Monte Carlo dropout is a technique for uncertainty quantification in neural networks. During inference, instead of turning off dropout (as is typical), you keep dropout active and generate multiple predictions with different random dropout patterns. Each prediction represents a slightly different version of the model, and the variance across these predictions provides an estimate of model uncertainty. This is particularly valuable in time series forecasting where understanding prediction confidence is crucial for business decision-making. For example, if the model predicts sales of 1000 units with high uncertainty, you might prepare for a wider range of outcomes than if the same prediction had low uncertainty. This uncertainty information helps in risk management and inventory planning.

---

### 8. In Prophet, what role do changepoints serve in the forecasting model?
**Answer: b) Changepoints identify points where the trend rate changes significantly**

**Explanation:** Changepoints in Prophet are specific dates where the growth rate of the trend component changes. Prophet automatically detects these points where the time series behavior shifts - for example, when a company's growth accelerates due to a new product launch or slows due to market saturation. By modeling these changepoints, Prophet can capture non-linear trends that would be missed by simple linear trend models. The model places changepoints at regular intervals and uses a sparse prior to determine which ones are significant. This allows Prophet to adapt to structural changes in the business while avoiding overfitting to noise. Changepoint detection is crucial for accurate long-term forecasting in dynamic business environments.

---

### 9. When combining forecasts in an ensemble, what is the advantage of weighted averaging over simple averaging?
**Answer: b) Weighted averaging allows better-performing models to have more influence on the final forecast**

**Explanation:** Weighted averaging recognizes that not all models in an ensemble perform equally well, and gives more influence to models that have demonstrated better performance on validation data. Simple averaging treats all models equally, which means a poorly performing model has the same impact as a high-performing model. Weighted averaging can use various schemes: inverse error weighting (models with lower errors get higher weights), performance-based ranking, or dynamic weighting based on recent performance. This leads to better ensemble performance because the final forecast is dominated by the most reliable models. In production systems, weights can be updated over time as model performance changes, creating an adaptive ensemble that maintains high accuracy as conditions evolve.

---

### 10. What is the most critical consideration when implementing time series forecasting in production?
**Answer: b) Ensuring consistent data preprocessing and handling of missing values across training and serving**

**Explanation:** Training-serving skew is one of the most common causes of production ML failures. In time series forecasting, this means ensuring that data preprocessing (scaling, differencing, missing value imputation, feature engineering) is identical between training and production serving. Any inconsistency can cause significant performance degradation or complete model failure. For example, if you scale data using training statistics but production data has a different distribution, or if you handle missing values differently in training versus serving, the model will receive inputs it wasn't trained on. Production systems must implement robust data pipelines with consistent preprocessing, proper handling of edge cases (missing data, outliers, data quality issues), and monitoring to detect when input data distributions change. This operational excellence is more critical than model complexity for production success.
