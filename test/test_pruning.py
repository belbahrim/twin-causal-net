import pytest
import twincausal
import pickle

with open('best_qini_max_prune_test.pickle', 'rb') as handle:
    best_qini = pickle.load(handle)

best_qini_train = best_qini['best_qini_train']
best_qini_test = best_qini['best_qini_test']
best_qini_oos = best_qini['best_qini_oos']

@pytest.mark.parametrize("results_obtained, expected_result", [
    (round(best_qini_oos,3),3.711),
    (round(best_qini_test,3),3.183),
    (round(best_qini_train,3),3.597)
])

def test_prunning_false(results_obtained, expected_result):
    assert results_obtained == expected_result


