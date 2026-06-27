# Monitoring Plan — Recommendation API

## Service-Level Objectives (SLOs)

| Metric | Target | Alert threshold |
|--------|--------|-----------------|
| Availability | 99.9% uptime | < 99.5% over 1 hour |
| Latency (P95) | < 150ms | > 200ms over 5 min |
| Latency (P99) | < 300ms | > 500ms over 5 min |
| Error rate | < 0.5% | > 1% over 5 min |
| Recommendation PSI | < 0.1 | > 0.2 (indicates drastic shift in recs) |

## Metrics Collected

### Application Metrics (Prometheus)

| Metric | Type | Description |
|--------|------|-------------|
| `rec_api_requests_total` | Counter | Total requests by endpoint and status code |
| `rec_api_request_duration_seconds` | Histogram | Request latency distribution |
| `rec_api_prediction_score` | Histogram | Distribution of top recommendation scores/probabilities |
| `rec_api_item_popularity_total` | Counter | Frequency of items being recommended (tracks popularity bias) |
| `rec_api_model_info` | Info | Model version, registry tag, and architecture |

### Infrastructure Metrics

| Metric | Source | Alert condition |
|--------|--------|-----------------|
| Pod CPU usage | K8s Metrics | > 70% sustained |
| Pod Memory usage | K8s Metrics | > 80% sustained |
| Error Logs (5xx) | Application Logs | > 10 in 1 min |

## Drift Detection Strategy

| Check | Frequency | Method | Action if triggered |
|-------|-----------|--------|---------------------|
| Interaction Volume | Daily | Statistical check | Investigate source data pipeline |
| Feature Stability (Users) | Weekly | PSI | Review user profile feature engineering |
| Recommendation Diversity | Daily | Gini Coefficient | Investigate if model is collapsing to popular items |
| Online vs Offline Gap | Monthly | A/B testing or Shadowing | Re-calibrate model parameters |

## Alerting

| Severity | Condition | Response |
|----------|-----------|----------|
| **P0 (Critical)** | API down or > 50% error rate | Immediate page, automated rollback to previous version |
| **P1 (High)** | Latency P99 > 1s | Investigate resource saturation or DB bottlenecks |
| **P2 (Medium)** | Diversity index drops > 30% | Review model training for popularity bias |
| **P3 (Low)** | New item coverage < 10% | Adjust exploration strategy in next training run |

## Dashboard Panels (Grafana)

1. **Throughput:** RPS (Requests Per Second)
2. **Success Rate:** % 2xx vs 4xx/5xx
3. **Latency Heatmap:** Response time distribution
4. **Top Recommended Items:** Table/Bar chart of the most frequently recommended product IDs
5. **Diversity Index:** Time series of Gini coefficient or unique item count in Top-K
6. **Model Version:** Current deployed tag from registry
