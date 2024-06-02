import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.dotQuery import dotQuery
from modules.getConn import getConections

class TestDotQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn, cls.cur = getConections()
        
    def test_dotQuery(self):
        test_cases = [
            {"inputText": "RED", "topk": 2, "includeDistance": False, "includeVector": False, "metadataFilter": None, "inId": None},
            {"inputText": "RED", "topk": 2, "includeDistance": False, "includeVector": False, "metadataFilter": ['ts'], "inId": None},
            {"inputText": "RED", "topk": 2, "includeDistance": False, "includeVector": False, "metadataFilter": None, "inId": ['1716655832.8881412']},
            {"inputText": "RED", "topk": 2, "includeDistance": False, "includeVector": False, "metadataFilter": ['ts'], "inId": ['1716655832.8881412']},

            {"inputText": "RED", "topk": 2, "includeDistance": True, "includeVector": False, "metadataFilter": None, "inId": None},
            {"inputText": "RED", "topk": 2, "includeDistance": True, "includeVector": False, "metadataFilter": ['ts'], "inId": None},
            {"inputText": "RED", "topk": 2, "includeDistance": True, "includeVector": False, "metadataFilter": None, "inId": ['1716655832.8881412']},
            {"inputText": "RED", "topk": 2, "includeDistance": True, "includeVector": False, "metadataFilter": ['ts'], "inId": ['1716655832.8881412']},

            {"inputText": "RED", "topk": 2, "includeDistance": False, "includeVector": True, "metadataFilter": None, "inId": None},
            {"inputText": "RED", "topk": 2, "includeDistance": False, "includeVector": True, "metadataFilter": ['ts'], "inId": None},
            {"inputText": "RED", "topk": 2, "includeDistance": False, "includeVector": True, "metadataFilter": None, "inId": ['1716655832.8881412']},
            {"inputText": "RED", "topk": 2, "includeDistance": False, "includeVector": True, "metadataFilter": ['ts'], "inId": ['1716655832.8881412']},

            {"inputText": "RED", "topk": 2, "includeDistance": True, "includeVector": True, "metadataFilter": None, "inId": None},
            {"inputText": "RED", "topk": 2, "includeDistance": True, "includeVector": True, "metadataFilter": ['ts'], "inId": None},
            {"inputText": "RED", "topk": 2, "includeDistance": True, "includeVector": True, "metadataFilter": None, "inId": ['1716655832.8881412']},
            {"inputText": "RED", "topk": 2, "includeDistance": True, "includeVector": True, "metadataFilter": ['ts'], "inId": ['1716655832.8881412']}
        ]
        
        for i, case in enumerate(test_cases):
            print(f"Running test case {i+1}")
            with self.subTest(i=i):
                result = dotQuery(
                    inputText=case["inputText"], 
                    topk=case["topk"], 
                    includeDistance=case["includeDistance"], 
                    includeVector=case["includeVector"], 
                    metadataFilter=case["metadataFilter"], 
                    inId=case["inId"]
                )
                self.assertIsNotNone(result)  # Check if result is not None

if __name__ == "__main__":
    unittest.main()
