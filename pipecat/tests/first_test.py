import unittest
from pipecat.services.rev_chatter import ReverieChatterLLMService

class TestReverieChatterLLMService(unittest.TestCase):
    def test_clean_phone_number(self):
        service = ReverieChatterLLMService(api_key="your_api_key")
        
        # Test case 1: Phone number with dashes
        phone_number = "123-456-7890"
        cleaned_number = service._clean_phone_number(phone_number)
        self.assertEqual(cleaned_number, "1234567890")
        
        # Test case 2: Phone number with spaces
        phone_number = "123 456 7890"
        cleaned_number = service._clean_phone_number(phone_number)
        self.assertEqual(cleaned_number, "1234567890")
        
        # Test case 3: Phone number with mixed dashes and spaces
        phone_number = "123-456 7890"
        cleaned_number = service._clean_phone_number(phone_number)
        self.assertEqual(cleaned_number, "1234567890")
        
        # Test case 4: Phone number with parentheses and dashes
        phone_number = "(123) 456-7890"
        cleaned_number = service._clean_phone_number(phone_number)
        self.assertEqual(cleaned_number, "1234567890")
        
        # Test case 5: Phone number with non-digit characters
        phone_number = "123-456-7890 ext. 123"
        cleaned_number = service._clean_phone_number(phone_number)
        self.assertEqual(cleaned_number, "1234567890")
        
        # Test case 6: Phone number with invalid format
        phone_number = "1234"
        cleaned_number = service._clean_phone_number(phone_number)
        self.assertEqual(cleaned_number, "1234")
        
        # Test case 7: Phone number with no formatting
        phone_number = "1234567890"
        cleaned_number = service._clean_phone_number(phone_number)
        self.assertEqual(cleaned_number, "1234567890")

if __name__ == "__main__":
    unittest.main()