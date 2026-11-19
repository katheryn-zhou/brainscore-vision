import pytest
import brainscore_vision


@pytest.mark.parametrize('identifier', ['shallow_e2e_conv4_w1_s0'])
def test_has_identifier(identifier):
    model = brainscore_vision.load_model(identifier)
    assert model.identifier == identifier

# @pytest.mark.travis_slow
# def test_has_identifier():
#     model = brainscore_vision.load_model('shallow_e2e_conv4_w1_s0')
#     assert model.identifier == 'shallow_e2e_conv4_w1_s0'