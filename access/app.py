"""Streamlit app to visualize generated vs reference cascade trees."""
import json
import streamlit as st


def build_tree_html(times, depths, parents, label, color_node, color_edge):
    """Build an HTML/CSS tree visualization from flat event list."""
    n = len(times)
    if n == 0:
        return f'<div style="color:#888;padding:16px;">No events in {label}</div>'

    children = {i: [] for i in range(-1, n)}
    for i, p in enumerate(parents):
        children[p].append(i)

    depth_colors = ['#4A90D9', '#50C878', '#F5A623', '#D94A7A', '#9B59B6', '#1ABC9C']

    def render_node(idx):
        t = times[idx]
        d = depths[idx] if idx < len(depths) else 0
        dc = depth_colors[d % len(depth_colors)]
        time_str = f'{t:.4f}' if t < 1 else f'{t:.1f}'
        badge = f'd={d}'
        node_html = (
            f'<div style="display:inline-block;background:{dc};color:#fff;'
            f'border-radius:6px;padding:3px 10px;margin:2px 0;font-size:13px;'
            f'font-family:monospace;cursor:default;" '
            f'title="time={t:.6f}, depth={d}, parent_idx={parents[idx]}">'
            f'<b>#{idx}</b> t={time_str} <span style="opacity:0.8">{badge}</span>'
            f'</div>'
        )
        kids = children.get(idx, [])
        if not kids:
            return f'<div style="margin-left:0">{node_html}</div>'

        kids_html = ''.join(render_node(c) for c in kids)
        return (
            f'<div style="margin-left:0">{node_html}'
            f'<div style="margin-left:24px;border-left:2px solid {color_edge};padding-left:8px;">'
            f'{kids_html}</div></div>'
        )

    roots = children.get(-1, [])
    if not roots:
        roots = [0]

    tree_body = ''.join(render_node(r) for r in roots)

    return (
        f'<div style="background:#1a1a2e;border-radius:10px;padding:16px;margin:8px 0;">'
        f'<div style="color:{color_node};font-weight:bold;font-size:15px;margin-bottom:8px;">'
        f'{label} ({n} events)</div>'
        f'{tree_body}</div>'
    )


def build_timeline_svg(times_gen, times_ref, width=700, height=80):
    """Build a simple SVG timeline comparing gen vs ref event times."""
    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    svg += f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0e0e1a" rx="8"/>'

    # Axis
    y_gen, y_ref = 25, 55
    svg += f'<line x1="40" y1="{y_gen}" x2="{width-20}" y2="{y_gen}" stroke="#444" stroke-width="1"/>'
    svg += f'<line x1="40" y1="{y_ref}" x2="{width-20}" y2="{y_ref}" stroke="#444" stroke-width="1"/>'
    svg += f'<text x="4" y="{y_gen+4}" fill="#50C878" font-size="11" font-family="monospace">Gen</text>'
    svg += f'<text x="4" y="{y_ref+4}" fill="#4A90D9" font-size="11" font-family="monospace">Ref</text>'

    all_t = list(times_gen) + list(times_ref)
    if not all_t:
        svg += '</svg>'
        return svg
    t_max = max(max(all_t), 1e-6)
    scale = (width - 60) / t_max

    for t in times_gen:
        x = 40 + t * scale
        svg += f'<circle cx="{x:.1f}" cy="{y_gen}" r="3" fill="#50C878" opacity="0.85"/>'
    for t in times_ref:
        x = 40 + t * scale
        svg += f'<circle cx="{x:.1f}" cy="{y_ref}" r="2" fill="#4A90D9" opacity="0.6"/>'

    svg += '</svg>'
    return svg


def main():
    st.set_page_config(page_title='Cascade Tree Viewer', layout='wide')

    st.markdown(
        '<h1 style="text-align:center;">Cascade Tree Viewer</h1>'
        '<p style="text-align:center;color:#888;">Compare generated vs reference cascade trees</p>',
        unsafe_allow_html=True,
    )

    default_path = '/scratch/gw2556/ODE/CasGen/runs/1860714/eval/generated_cascades.json'
    path = st.text_input('Path to generated_cascades.json', value=default_path)

    if not path:
        st.info('Enter a path above to load data.')
        return

    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f'File not found: {path}')
        return
    except json.JSONDecodeError:
        st.error(f'Invalid JSON: {path}')
        return

    st.success(f'Loaded {len(data)} cascades')

    # Detect format
    has_ref = 'ref_times' in data[0]

    # Stats sidebar
    with st.sidebar:
        st.markdown('### Stats')
        gen_key = 'gen_times' if has_ref else 'times'
        gen_lens = [len(c[gen_key]) for c in data]
        st.metric('Total cascades', len(data))
        st.metric('Avg gen events', f'{sum(gen_lens)/len(gen_lens):.1f}')
        if has_ref:
            ref_lens = [len(c['ref_times']) for c in data]
            st.metric('Avg ref events', f'{sum(ref_lens)/len(ref_lens):.1f}')

        # Metrics file
        import os
        metrics_path = os.path.join(os.path.dirname(path), 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            st.markdown('### Metrics')
            st.metric('MMD ↓', f'{metrics["mmd"]:.4f}')
            w1_key = 'wasserstein_count' if 'wasserstein_count' in metrics else 'w1_count'
            if w1_key in metrics:
                st.metric('Wasserstein ↓', f'{metrics[w1_key]:.4f}')

        st.markdown('### Filter')
        min_events = st.slider('Min gen events', 0, max(gen_lens) if gen_lens else 10, 0)

    # Filter cascades
    filtered = []
    for i, c in enumerate(data):
        gk = 'gen_times' if has_ref else 'times'
        if len(c[gk]) >= min_events:
            filtered.append((i, c))

    st.markdown(f'**Showing {len(filtered)} / {len(data)} cascades** (min events ≥ {min_events})')

    if not filtered:
        st.warning('No cascades match the filter.')
        return

    # Cascade selector
    labels = [f'#{idx} — gen:{len(c["gen_times" if has_ref else "times"])} events'
              + (f', ref:{len(c["ref_times"])} events' if has_ref else '')
              for idx, c in filtered]

    selected = st.selectbox('Select cascade', range(len(filtered)), format_func=lambda i: labels[i])
    idx, cascade = filtered[selected]

    st.markdown(f'---')
    st.markdown(f'#### Cascade #{idx}')

    if has_ref:
        st.markdown('**Timeline** (green=generated, blue=reference)')
        svg = build_timeline_svg(cascade['gen_times'], cascade['ref_times'])
        st.markdown(svg, unsafe_allow_html=True)

        has_tree = 'gen_depths' in cascade and 'gen_tree_parents' in cascade

        col1, col2 = st.columns(2)
        with col1:
            if has_tree:
                html = build_tree_html(
                    cascade['gen_times'], cascade['gen_depths'],
                    cascade['gen_tree_parents'],
                    'Generated Tree', '#50C878', '#2d6b3f',
                )
            else:
                n_gen = cascade.get('gen_n', len(cascade['gen_times']))
                html = build_tree_html(
                    cascade['gen_times'], [0] * n_gen,
                    list(range(-1, n_gen - 1)),
                    f'Generated ({n_gen} events)', '#50C878', '#2d6b3f',
                )
            st.markdown(html, unsafe_allow_html=True)

        with col2:
            if 'ref_depths' in cascade and 'ref_tree_parents' in cascade:
                html = build_tree_html(
                    cascade['ref_times'], cascade['ref_depths'],
                    cascade['ref_tree_parents'],
                    'Reference Tree', '#4A90D9', '#2a5a8c',
                )
            else:
                n_ref = cascade.get('ref_n', len(cascade['ref_times']))
                html = build_tree_html(
                    cascade['ref_times'], [0] * n_ref,
                    list(range(-1, n_ref - 1)),
                    f'Reference ({n_ref} events)', '#4A90D9', '#2a5a8c',
                )
            st.markdown(html, unsafe_allow_html=True)
    else:
        depths = cascade.get('depths', [0] * len(cascade['times']))
        parents = cascade.get('tree_parents', list(range(-1, len(cascade['times']) - 1)))
        html = build_tree_html(
            cascade['times'], depths, parents,
            'Generated Tree', '#50C878', '#2d6b3f',
        )
        st.markdown(html, unsafe_allow_html=True)

    # Raw data expander
    with st.expander('Raw JSON'):
        st.json(cascade)


if __name__ == '__main__':
    main()
