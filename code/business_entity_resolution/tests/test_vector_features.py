import numpy as np
from ber.vector_features import cosine_matrix_rows, dequantize, pair_vector_features, quantize


def test_pair_vector_features_columns():
    v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    f = pair_vector_features(np.array([0, 1]), np.array([1, 0]), v, v)
    assert f.shape == (2, 3)
    assert np.allclose(f[:, 0], 0.0)
    f2 = pair_vector_features(np.array([0, 1]), np.array([0, 1]), v, v)
    assert np.allclose(f2[:, 0], 1.0)


def test_cosine_matches_manual():
    a = np.array([[1.0, 0.0]], dtype="float32")
    b = np.array([[1.0, 1.0]], dtype="float32")
    assert abs(cosine_matrix_rows(a, b)[0] - 0.7071067) < 1e-5


def test_quantize_roundtrip_close():
    v = np.array([[0.5, -0.25, 1.0]], dtype="float32")
    assert np.allclose(dequantize(quantize(v)), v, atol=1 / 127 + 1e-6)
