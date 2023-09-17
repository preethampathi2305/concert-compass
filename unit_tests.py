import numpy as np
from ml import ml


def test_pusha_t():
    rec_out = ml("spotify:playlist:37i9dQZF1EQnqst5TRi17F")
    rec_dict = {}
    for key, value in rec_out.items():
        temp = np.array(value)
        for i in temp:
            rec_dict[i[0]] = i[1]
    assert "Pusha T" in rec_dict.keys()
    assert rec_dict["Pusha T"] >= 0.0


def test_suicideboy():
    rec_out = ml("spotify:playlist:37i9dQZF1EQnqst5TRi17F")
    rec_dict = {}
    for key, value in rec_out.items():
        temp = np.array(value)
        for i in temp:
            rec_dict[i[0]] = i[1]
    assert "$uicideboy$" in rec_dict.keys()
    assert rec_dict["$uicideboy$"] >= 0.0


def test_saba():
    rec_out = ml("spotify:playlist:37i9dQZF1EQnqst5TRi17F")
    rec_dict = {}
    for key, value in rec_out.items():
        temp = np.array(value)
        for i in temp:
            rec_dict[i[0]] = i[1]
    assert "Saba" in rec_dict.keys()
    assert rec_dict["Saba"] >= 0.0
