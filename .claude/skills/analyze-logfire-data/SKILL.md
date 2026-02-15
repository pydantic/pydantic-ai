---
name: analyze-logfire-data
description: Query Logfire database and generate rich dashboards from results. Use when analyzing telemetry data, creating visualizations, or exploring Logfire records.
argument-hint: '[analysis goal]'
---

# Analyze Logfire Data

Query the Logfire database and generate interactive dashboards from telemetry data.

## Prerequisites

- `LOGFIRE_READ_TOKEN` environment variable must be set
- Run queries using: `uv run --with logfire python logfire_query.py`

## Workflow

### 1. Understand the Schema

First, use the `mcp__logfire__schema_reference` tool to understand the database schema. Key tables and columns:

- `records` table contains spans and logs
- Important columns: `message`, `span_name`, `trace_id`, `exception_type`, `exception_message`, `start_timestamp`, `service_name`, `attributes`
- Use `->` and `->>` operators for JSON fields in `attributes`

### 2. Query the Data

Use `logfire_query.py` from this skill directory to execute queries:

```python
from logfire_query import query_sync, load_results

# Execute query and save results
query_sync(
    '''
    SELECT span_name, count(*) as count
    FROM records
    WHERE start_timestamp > now() - INTERVAL '1 hour'
    GROUP BY span_name
    ORDER BY count DESC
    LIMIT 20
    ''',
    'results.json'
)

# Load results for analysis
rows = load_results('results.json')
```

Alternatively, use the `mcp__logfire__arbitrary_query` tool directly for simpler queries.

### 3. Generate Dashboards

Write custom Plotly code for rich, interactive dashboards. Do NOT use predefined chart utilities - write the visualization code directly.

#### Dashboard Guidelines

**Structure:**
- Use `plotly.subplots.make_subplots()` for multi-panel layouts
- Include multiple visualization types: bar, line, scatter, heatmap, pie
- Save as HTML for full interactivity

**Styling:**
- Consistent color scheme (use `plotly.express.colors` palettes)
- Clear, descriptive titles for each subplot
- Proper axis labels with units
- Legend placement that doesn't obscure data

**Interactivity:**
- Enable hover tooltips with relevant data
- Support zoom/pan for time series
- Add range sliders for date filtering where appropriate

#### Example Dashboard Code

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Create multi-panel dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Requests by Endpoint', 'Error Rate Over Time',
                    'Response Time Distribution', 'Top Error Types'),
    specs=[[{'type': 'bar'}, {'type': 'scatter'}],
           [{'type': 'histogram'}, {'type': 'pie'}]]
)

# Add traces to each subplot
fig.add_trace(
    go.Bar(x=endpoints, y=counts, marker_color=px.colors.qualitative.Set2),
    row=1, col=1
)

# ... add more traces ...

fig.update_layout(
    height=800,
    showlegend=True,
    title_text='Logfire Telemetry Dashboard',
    title_x=0.5
)

fig.write_html('dashboard.html')
print('Dashboard saved to dashboard.html')
```

## Common Query Patterns

### Find Exceptions
```sql
SELECT exception_type, exception_message, count(*) as count
FROM records
WHERE exception_type IS NOT NULL
  AND start_timestamp > now() - INTERVAL '24 hours'
GROUP BY exception_type, exception_message
ORDER BY count DESC
LIMIT 20
```

### Trace Latency Analysis
```sql
SELECT span_name,
       avg(duration) as avg_duration,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY duration) as p95
FROM records
WHERE start_timestamp > now() - INTERVAL '1 hour'
GROUP BY span_name
ORDER BY avg_duration DESC
```

### Service Dependencies
```sql
SELECT service_name, span_name, count(*) as calls
FROM records
WHERE start_timestamp > now() - INTERVAL '1 hour'
GROUP BY service_name, span_name
ORDER BY calls DESC
```

## Output

- Save dashboards as `.html` files for interactivity
- Include the file path in your response so the user can open it
- For quick insights, print summary statistics to stdout
