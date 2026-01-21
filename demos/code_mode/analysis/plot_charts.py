# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "plotly",
#     "pandas",
# ]
# ///
"""Generate comparison charts from fetched Logfire data.

Usage:
    uv run python demos/code_mode/analysis/plot_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

COLORS = {'traditional': '#EBE6FD', 'code_mode': '#E520E9'}
MODE_LABELS = {'traditional': 'Traditional', 'code_mode': 'Code Mode'}


def load_data(filename: str) -> pd.DataFrame:
    """Load JSON data into a DataFrame."""
    with open(DATA_DIR / filename) as f:
        data = json.load(f)
    return pd.DataFrame(data['rows'])


def create_latency_chart():
    """Create latency comparison chart."""
    df = load_data('latency.json')
    df['mode_label'] = df['mode'].map(MODE_LABELS)

    # Box plot for distribution
    fig = px.box(
        df,
        x='mode_label',
        y='duration_seconds',
        color='mode',
        color_discrete_map=COLORS,
        title='Latency Comparison: Traditional vs Code Mode',
        labels={'duration_seconds': 'Duration (seconds)', 'mode_label': 'Mode'},
    )
    fig.update_layout(showlegend=False)
    fig.write_html(RESULTS_DIR / 'latency_comparison.html')

    # Summary stats
    summary = df.groupby('mode')['duration_seconds'].agg(['mean', 'std', 'min', 'max'])
    print('Latency Summary (seconds):')
    print(summary.round(2))
    print()


def create_token_chart():
    """Create token usage comparison chart."""
    df = load_data('tokens.json')
    df['total_tokens'] = df['input_tokens'] + df['output_tokens']
    df['mode_label'] = df['mode'].map(MODE_LABELS)

    # Grouped bar chart for input/output tokens
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Input Tokens', 'Output Tokens'])

    for i, token_type in enumerate(['input_tokens', 'output_tokens'], 1):
        summary = df.groupby('mode')[token_type].mean().reset_index()
        summary['mode_label'] = summary['mode'].map(MODE_LABELS)

        fig.add_trace(
            go.Bar(
                x=summary['mode_label'],
                y=summary[token_type],
                marker_color=[COLORS[m] for m in summary['mode']],
                name=token_type.replace('_', ' ').title(),
            ),
            row=1,
            col=i,
        )

    fig.update_layout(
        title='Token Usage Comparison: Traditional vs Code Mode',
        showlegend=False,
    )
    fig.write_html(RESULTS_DIR / 'token_comparison.html')

    # Summary stats
    summary = df.groupby('mode')[['input_tokens', 'output_tokens', 'total_tokens']].mean()
    print('Token Usage Summary (average per run):')
    print(summary.round(0).astype(int))
    print()


def create_request_count_chart():
    """Create request count comparison chart."""
    df = load_data('requests.json')
    df['mode_label'] = df['mode'].map(MODE_LABELS)

    fig = px.bar(
        df.groupby('mode')['request_count'].mean().reset_index(),
        x=df.groupby('mode')['request_count'].mean().reset_index()['mode'].map(MODE_LABELS),
        y='request_count',
        color=df.groupby('mode')['request_count'].mean().reset_index()['mode'],
        color_discrete_map=COLORS,
        title='LLM Requests per Run: Traditional vs Code Mode',
        labels={'x': 'Mode', 'request_count': 'Average Requests'},
    )
    fig.update_layout(showlegend=False)
    fig.write_html(RESULTS_DIR / 'request_comparison.html')

    # Summary
    summary = df.groupby('mode')['request_count'].agg(['mean', 'std'])
    print('Request Count Summary:')
    print(summary.round(2))
    print()


def create_cost_chart():
    """Create cost comparison chart."""
    df = load_data('cost.json')
    df['mode_label'] = df['mode'].map(MODE_LABELS)

    fig = px.bar(
        df.groupby('mode')['total_cost'].mean().reset_index(),
        x=df.groupby('mode')['total_cost'].mean().reset_index()['mode'].map(MODE_LABELS),
        y='total_cost',
        color=df.groupby('mode')['total_cost'].mean().reset_index()['mode'],
        color_discrete_map=COLORS,
        title='Cost per Run: Traditional vs Code Mode',
        labels={'x': 'Mode', 'total_cost': 'Average Cost ($)'},
    )
    fig.update_layout(showlegend=False)
    fig.write_html(RESULTS_DIR / 'cost_comparison.html')

    # Summary
    summary = df.groupby('mode')['total_cost'].agg(['mean', 'sum'])
    print('Cost Summary ($):')
    print(summary.round(4))
    print()


def create_summary_dashboard():
    """Create a combined dashboard with all metrics."""
    latency_df = load_data('latency.json')
    tokens_df = load_data('tokens.json')
    requests_df = load_data('requests.json')
    cost_df = load_data('cost.json')

    # Calculate summaries
    latency_summary = latency_df.groupby('mode')['duration_seconds'].mean()
    tokens_df['total_tokens'] = tokens_df['input_tokens'] + tokens_df['output_tokens']
    tokens_summary = tokens_df.groupby('mode')['total_tokens'].mean()
    requests_summary = requests_df.groupby('mode')['request_count'].mean()
    cost_summary = cost_df.groupby('mode')['total_cost'].mean()

    # Create dashboard
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=['Latency (seconds)', 'Total Tokens', 'LLM Requests', 'Cost ($)'],
        specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]],
    )

    modes = ['traditional', 'code_mode']
    labels = [MODE_LABELS[m] for m in modes]
    colors = [COLORS[m] for m in modes]

    # Latency
    fig.add_trace(go.Bar(x=labels, y=[latency_summary[m] for m in modes], marker_color=colors), row=1, col=1)

    # Tokens
    fig.add_trace(go.Bar(x=labels, y=[tokens_summary[m] for m in modes], marker_color=colors), row=1, col=2)

    # Requests
    fig.add_trace(go.Bar(x=labels, y=[requests_summary[m] for m in modes], marker_color=colors), row=2, col=1)

    # Cost
    fig.add_trace(go.Bar(x=labels, y=[cost_summary[m] for m in modes], marker_color=colors), row=2, col=2)

    fig.update_layout(
        title='Code Mode vs Traditional: Performance Comparison',
        showlegend=False,
        height=600,
        width=800,
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    fig.write_html(RESULTS_DIR / 'dashboard.html')

    # Also export as PNG for README embedding
    try:
        fig.write_image(RESULTS_DIR / 'dashboard.png', scale=2)
        print('  Exported dashboard.png')
    except Exception as e:
        print(f'  Note: Could not export PNG (install kaleido): {e}')

    # Print savings
    print('=' * 60)
    print('SAVINGS SUMMARY')
    print('=' * 60)
    for name, trad, code in [
        ('Latency', latency_summary['traditional'], latency_summary['code_mode']),
        ('Tokens', tokens_summary['traditional'], tokens_summary['code_mode']),
        ('Requests', requests_summary['traditional'], requests_summary['code_mode']),
        ('Cost', cost_summary['traditional'], cost_summary['code_mode']),
    ]:
        savings = (trad - code) / trad * 100
        print(f'{name}: {savings:.1f}% reduction with Code Mode')
    print('=' * 60)


def main():
    print('Generating charts...')
    print()

    create_latency_chart()
    create_token_chart()
    create_request_count_chart()
    create_cost_chart()
    create_summary_dashboard()

    print()
    print(f'Charts saved to {RESULTS_DIR}/')
    print()
    print('Open dashboard.html in your browser to view all metrics.')


if __name__ == '__main__':
    main()
