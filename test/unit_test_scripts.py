import unittest
import twincausal
import pickle

with open('best_qini.pickle', 'rb') as handle:
    best_qini = pickle.load(handle)

best_qini_train = best_qini['best_qini_train']
best_qini_test = best_qini['best_qini_test']
best_qini_oos = best_qini['best_qini_oos']



class Testtwincausal(unittest.TestCase):
    def test_twincausal(self):
        self.assertAlmostEqual(round(best_qini_oos,3),2.568)
        self.assertAlmostEqual(round(best_qini_test,3),3.027)
        self.assertAlmostEqual(round(best_qini_train,3),3.533)




    

        