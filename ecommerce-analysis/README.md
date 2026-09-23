# Bilan Data Cleaning

## What we have now

| Element                      |                             Value |
| ---------------------------- | --------------------------------: |
| Final rows                   |                           110,181 |
| Columns                      |                                21 |
| Delay rate (`est_en_retard`) |                              6.8% |
| Median distance              |                            432 km |
| Multivariate anomalies       | 2.0% (feature kept, not excluded) |
| Physical errors removed      |                                 8 |

The Data Cleaning milestone is officially complete. We now have a clean analytical table, a clear target variable, and geographically and logistically consistent features.

# Bilan EDA

## What we have now

| Element                    |                                                                        Value |
| -------------------------- | ---------------------------------------------------------------------------: |
| Main focus                 |                                              Exploratory Data Analysis (EDA) |
| Data quality status        |                                                 Clean and ready for analysis |
| Key variables studied      | Order date, delivery delay, payment, product category, seller zone, distance |
| Notable patterns           | Strong concentration of late deliveries in certain cities and product groups |
| Customer behavior insights |           Orders are mostly concentrated in a few key regions and categories |
| Operational signals        |     Logistics and distance variables show meaningful influence on delay risk |
| Recommendation status      |               Insights are ready to support feature engineering and modeling |

The EDA milestone is complete. We have explored the structure, quality, and relationships in the dataset to better understand the business drivers behind delivery performance and customer behavior.

## What we have done in this milestone

- We reviewed the cleaned dataset to understand its overall structure, distributions, and missing-value patterns.
- We analyzed the target variable and identified the main drivers of delivery delay, including logistics and geographic factors.
- We explored customer, product, and seller behavior to detect repeat patterns and segmentation opportunities.
- We examined correlations and distributions for key variables such as distance, order value, category, and delivery timing.
- We visualized the main anomalies and outliers to verify whether they were meaningful business signals or data issues.
- We extracted actionable insights to guide feature engineering and the machine learning phase.
- We validated that the business story is coherent before moving to predictive modeling.
