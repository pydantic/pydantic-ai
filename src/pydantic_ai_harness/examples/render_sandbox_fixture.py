"""Owned invoice fixture; acceptance checks stay out of the agent workspace."""

SOURCE = b'''def invoice_total(lines):
    """Sum unit_price_cents * quantity for each invoice line."""
    return sum(line["unit_price_cents"] for line in lines)
'''

VISIBLE_TESTS = b"""import unittest
from totals import invoice_total

class InvoiceTests(unittest.TestCase):
    def test_quantity(self):
        self.assertEqual(invoice_total([{"unit_price_cents": 125, "quantity": 3}]), 375)

    def test_empty(self):
        self.assertEqual(invoice_total([]), 0)

if __name__ == "__main__":
    unittest.main()
"""

HELD_OUT_TESTS = b"""import unittest
from totals import invoice_total

class AcceptanceTests(unittest.TestCase):
    def test_zero_quantity(self):
        self.assertEqual(invoice_total([{"unit_price_cents": 200, "quantity": 0}]), 0)

    def test_multiple_lines(self):
        self.assertEqual(invoice_total([{"unit_price_cents": 1299, "quantity": 2},
            {"unit_price_cents": 499, "quantity": 3}]), 4095)

    def test_zero_price(self):
        self.assertEqual(invoice_total([{"unit_price_cents": 0, "quantity": 100}]), 0)

    def test_large_exact_integer(self):
        self.assertEqual(invoice_total([{"unit_price_cents": 9007199254740993,
            "quantity": 7}]), 63050394783186951)

    def test_no_mutation(self):
        data = [{"unit_price_cents": 9, "quantity": 4}]
        self.assertEqual(invoice_total(data), 36)
        self.assertEqual(data, [{"unit_price_cents": 9, "quantity": 4}])

if __name__ == "__main__":
    unittest.main()
"""
