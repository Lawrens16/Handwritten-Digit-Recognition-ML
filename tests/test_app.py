from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_training_and_selection_preserve_results():
    assert Path('app.py').exists(), 'three-tab application is missing'
    app = AppTest.from_file(Path('app.py').resolve(), default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 3
    assert all(key not in app.session_state for key in ('classifier', 'clusters', 'agent'))
    for key, result_key in [('train_classifier', 'classifier'), ('train_clusters', 'clusters'), ('train_agent', 'agent')]:
        app.button(key=key).click().run()
        assert not app.exception
        assert result_key in app.session_state
    classifier = app.session_state['classifier']
    clusters = app.session_state['clusters']
    agent = app.session_state['agent']
    app.slider(key='sample').set_value(12).run()
    app.selectbox(key='cluster').select(4).run()
    assert not app.exception
    assert app.session_state['classifier'] is classifier
    assert app.session_state['clusters'] is clusters
    assert app.session_state['agent'] is agent


def test_confusion_cell_inspection_matches_images_without_retraining():
    import numpy as np
    from supervised import train_classifier
    app = AppTest.from_file(Path('app.py').resolve(), default_timeout=60)
    result = train_classifier()
    app.session_state['classifier'] = result
    app.run()
    assert not app.exception
    assert any(widget.key == 'actual_digit' for widget in app.selectbox), 'confusion-cell selectors missing'
    # Select a genuine off-diagonal error from the independently checked matrix.
    actual, predicted = np.argwhere(result.confusion - np.diag(np.diag(result.confusion)) > 0)[0]
    app.selectbox(key='actual_digit').select(int(actual)).run()
    app.selectbox(key='predicted_digit').select(int(predicted)).run()
    expected = np.flatnonzero((result.test_labels == actual) & (result.predictions == predicted))
    table = app.dataframe[0].value
    assert table['Held-out sample index'].tolist() == expected.tolist()
    assert table['Dataset index'].tolist() == result.test_indices[expected].tolist()
    assert table['Actual digit'].tolist() == [int(actual)] * len(expected)
    assert table['Predicted digit'].tolist() == [int(predicted)] * len(expected)
    app.selectbox(key='matching_sample').select(int(expected[-1])).run()
    assert app.session_state['classifier'] is result
    assert not app.exception
    # Empty cell must clear the image selector and table rather than show stale results.
    actual, predicted = np.argwhere(result.confusion == 0)[0]
    app.selectbox(key='actual_digit').select(int(actual)).run()
    app.selectbox(key='predicted_digit').select(int(predicted)).run()
    assert not app.exception
    assert not any(widget.key == 'matching_sample' for widget in app.selectbox)
    assert any('No held-out images' in info.value for info in app.info)
    assert app.session_state['classifier'] is result
