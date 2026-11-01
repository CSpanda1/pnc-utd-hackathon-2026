import unittest
from financial_tools import calculate_indicators,simulate_purchase
class Tests(unittest.TestCase):
    def test_cash_surplus(self):
        a=calculate_indicators('C001')
        self.assertAlmostEqual(a['monthly_cash_surplus'],round(a['monthly_income']-a['monthly_non_debt_expenses']-a['monthly_debt_payments'],2))
    def test_payment_reduces_surplus(self):
        a=calculate_indicators('C001');s=simulate_purchase('C001',30000,5000)
        self.assertAlmostEqual(s['projected_monthly_surplus'],round(a['monthly_cash_surplus']-s['new_monthly_payment'],2))
    def test_bad_down_payment(self):
        with self.assertRaises(ValueError):simulate_purchase('C001',1000,2000)
if __name__=='__main__':unittest.main()
