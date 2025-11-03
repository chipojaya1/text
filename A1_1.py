import re
import unittest
from typing import List

# Question 1: Validate an Email Address
def is_valid_email(email: str) -> bool:
    # TODO: write the regular expression below
    pattern = r''
    return bool(re.match(pattern, email))

# Question 2: Extract All Phone Numbers from a Text
def extract_phone_numbers(text: str) -> List[str]:
    # TODO: write the regular expression below
    pattern = r''
    return re.findall(pattern, text)

# Question 3: Replace All Digits with a Hash Symbol
def replace_digits(text: str) -> str:
    # TODO: write the regular expression below
    pattern = r''
    return re.sub(pattern, '#', text)

# Question 4: Find All Words Starting with a Capital Letter
def find_capital_words(text: str) -> List[str]:
    # TODO: write the regular expression below
    pattern = r''
    return re.findall(pattern, text)

# Question 5: Check if a String is a Valid Date (DD-MM-YYYY)
def is_valid_date(date_str: str) -> bool:
    # TODO: write the regular expression below
    pattern = r''
    return bool(re.match(pattern, date_str))

# Unit Tests
class TestRegexFunctions(unittest.TestCase):

    def test_is_valid_email(self):
        self.assertTrue(is_valid_email("user@example.com"))
        self.assertFalse(is_valid_email("userexample.com"))
        self.assertFalse(is_valid_email("user@.com"))

    def test_extract_phone_numbers(self):
        text = "Call 123-456-7890 or 987-654-3210."
        self.assertEqual(extract_phone_numbers(text), ["123-456-7890", "987-654-3210"])

    def test_replace_digits(self):
        self.assertEqual(replace_digits("My number is 12345"), "My number is #####")

    def test_find_capital_words(self):
        sentence = "Alice and Bob went to New York."
        self.assertEqual(find_capital_words(sentence), ["Alice", "Bob", "New", "York"])

    def test_is_valid_date(self):
        self.assertTrue(is_valid_date("15-09-2025"))
        self.assertFalse(is_valid_date("32-01-2025"))
        self.assertFalse(is_valid_date("15-13-2025"))

if __name__ == "__main__":
    unittest.main()