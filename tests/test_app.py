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
