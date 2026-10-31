"""Local three-paradigm learning demonstration. Run with Streamlit."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import load_digits

from supervised import train_classifier
from unsupervised import train_clusters
from reinforcement import make_environment, train_agent

st.set_page_config(page_title='Handwritten Digit Recognition', page_icon='🔢', layout='wide')
st.title('Handwritten Digit Recognition')
st.write('Explore three ways machines learn: labeled examples, visual similarity, and rewards.')
st.caption('Built-in 8×8 Digits dataset · Local CPU training · Seed 42')
recognize, group, navigate = st.tabs(['Recognize', 'Group', 'Navigate'])


def show_figure(figure):
    st.pyplot(figure, width='content')
    plt.close(figure)


with recognize:
    st.header('Learn from labeled examples')
    st.write('Logistic Regression learns a weighted score for each digit. The test images are held out during training.')
    if st.button('Train classifier', key='train_classifier', type='primary'):
        with st.spinner('Learning digit weights…'):
            st.session_state['classifier'] = train_classifier()
    if 'classifier' not in st.session_state:
        st.info('Train the classifier, then select a held-out image to inspect its prediction.')
    else:
        result = st.session_state['classifier']
        st.metric('Held-out accuracy', f'{result.accuracy:.2%}')
        st.caption(f'1,437 training images / 360 test images. Training and evaluation: {result.seconds:.2f} s.')
        sample = st.slider('Held-out sample index', 0, len(result.test_labels) - 1, 0, key='sample')
        left, right = st.columns([1, 2])
        with left:
            figure, axis = plt.subplots(figsize=(3, 3))
            axis.imshow(result.test_features[sample].reshape(8, 8), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axis.axis('off')
            show_figure(figure)
            prediction = int(result.model.predict(result.test_features[sample:sample + 1])[0])
            st.write(f'**Predicted digit: {prediction}** · Actual digit: {result.test_labels[sample]}')
        with right:
            figure, axis = plt.subplots(figsize=(6, 4))
            plot = axis.imshow(result.confusion, cmap='Blues')
            for row in range(10):
                for column in range(10):
                    count = result.confusion[row, column]
                    axis.text(column, row, str(count), ha='center', va='center', fontsize=8,
                              color='white' if count > result.confusion.max() / 2 else 'black')
            axis.set(xticks=range(10), yticks=range(10), xlabel='Predicted digit', ylabel='Actual digit', title='Confusion matrix: 360 held-out images')
            figure.colorbar(plot, ax=axis)
            figure.tight_layout()
            show_figure(figure)
        st.caption('fit() learns from training labels; predict() uses those weights on unseen test examples. This is not a neural network.')

with group:
    st.header('Find visual groups without labels')
    st.write('K-Means assigns each image to its nearest center, then updates the centers to the means of their members.')
    if st.button('Train clusters', key='train_clusters', type='primary'):
        with st.spinner('Finding ten clusters…'):
            st.session_state['clusters'] = train_clusters(load_digits().data / 16.0)
    if 'clusters' not in st.session_state:
        st.info('Train K-Means to see ten learned centers and inspect their members.')
    else:
        result = st.session_state['clusters']
        st.metric('Silhouette score', f'{result.silhouette:.4f}')
        st.caption(f'Clustering and silhouette calculation: {result.seconds:.2f} s. Silhouette measures cohesion/separation, not classification accuracy.')
        figure, axes = plt.subplots(2, 5, figsize=(10, 4))
        for cluster_id, axis in enumerate(axes.flat):
            axis.imshow(result.centers[cluster_id], cmap='gray', vmin=0, vmax=1)
            axis.set_title(f'Cluster {cluster_id} ({result.counts[cluster_id]})', fontsize=10)
            axis.axis('off')
        figure.tight_layout()
        show_figure(figure)
        st.warning('Cluster IDs are arbitrary: Cluster 3 does not mean digit 3. Groups may mix digits or split writing styles.')
        cluster = st.selectbox('Inspect cluster', range(10), key='cluster')
        members = np.flatnonzero(result.assignments == cluster)[:10]
        st.write(f'{result.counts[cluster]} members; showing the first {len(members)} in dataset order. Total across clusters: {result.counts.sum()}.')
        images = load_digits().images
        figure, axes = plt.subplots(1, 10, figsize=(10, 2))
        for axis in axes:
            axis.axis('off')
        for axis, member in zip(axes, members):
            axis.imshow(images[member], cmap='gray', vmin=0, vmax=16)
        figure.tight_layout()
        show_figure(figure)

with navigate:
    st.header('Learn actions through rewards')
    st.write('Q-Learning explores a separate navigation task: deterministic 4×4 FrozenLake. Reaching the goal gives +1; every other transition gives 0.')
    if st.button('Train agent', key='train_agent', type='primary'):
        with st.spinner('Training for 10,000 episodes…'):
            st.session_state['agent'] = train_agent()
    result = st.session_state.get('agent')
    environment = make_environment()
    tiles = environment.unwrapped.desc.astype(str)
    environment.close()
    figure, axis = plt.subplots(figsize=(4, 4))
    colors = {'S': '#b9d8ed', 'F': '#e5f2fa', 'H': '#39495c', 'G': '#9dd5b3'}
    names = {'S': 'Start', 'F': 'Ice', 'H': 'Hole', 'G': 'Goal'}
    for state in range(16):
        row, column = divmod(state, 4)
        tile = tiles[row, column]
        axis.add_patch(plt.Rectangle((column, row), 1, 1, facecolor=colors[tile], edgecolor='white', linewidth=2))
        axis.text(column + .5, row + .5, f'{state}\n{names[tile]}', ha='center', va='center', color='white' if tile == 'H' else '#16334a')
    if result:
        columns = [state % 4 + .5 for state in result.path]
        rows = [state // 4 + .5 for state in result.path]
        axis.plot(columns, rows, color='#c95021', linewidth=3, marker='o', alpha=.8)
    axis.set(xlim=(0, 4), ylim=(4, 0), aspect='equal')
    axis.axis('off')
    show_figure(figure)
    if not result:
        st.info('Train the agent to learn action values and display its greedy route.')
    else:
        st.metric('Greedy evaluation success', f'{result.success_rate:.0%} ({int(result.success_rate * 100)}/100)')
        st.caption(f'Training and evaluation: {result.seconds:.2f} s. Evaluation has no exploration or Q-table updates.')
        if result.status == 'goal':
            st.success('Learned route reaches the goal.')
        else:
            st.warning(f'Rollout ended: {result.status}.')
        st.write('Route (state IDs): ' + ' → '.join(map(str, result.path)))
        st.line_chart({'Episode': np.arange(100, 10001), 'Rolling goal rate': result.rolling_success}, x='Episode', y='Rolling goal rate')
        st.caption('Mean reward over the last 100 training episodes. Exploration starts at 1.0, decays by 0.999 per episode, and stops decaying at 0.05.')
        st.subheader('Learned Q-table')
        st.dataframe({name: result.q_table[:, index] for index, name in enumerate(['Left', 'Down', 'Right', 'Up'])})
        st.code('target = reward if terminated else reward + 0.95 * max(Q[next_state])\nQ[state, action] += 0.1 * (target - Q[state, action])', language='python')
        st.caption('Each row is a state ID (0–15). This evaluation repeats one map and start; it does not measure generalization to other maps.')
