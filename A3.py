# Required installations:
# pip install datasets
# For fasttext installation, refer to: https://fasttext.cc/docs/en/support.html

from datasets import load_dataset
import fasttext
import unittest
from typing import Dict, Union

# Utility function to map numeric labels to fastText format
def get_label_description(label: int) -> str:
    label_description: Dict[int, str] = {
        0: '__label__negative',
        1: '__label__positive'
    }
    if label not in label_description:
        raise KeyError(f"Invalid label: {label}")
    return label_description[label]

# Save IMDB dataset in FastText format
def save_dataset_in_fasttext_format() -> None:
    imdb = load_dataset('imdb')
    for split in ['test', 'train']:
        output_filename = f'imdb.{split}'
        with open(output_filename, 'w', encoding='utf-8') as f:
            for entry in imdb[split]:
                label = entry['label']
                text = entry['text'].replace('\n', ' ')  # Clean newlines
                label_desc = get_label_description(label)
                f.write(f'{label_desc}\t{text}\n')

# Train FastText supervised model
def train_model(input_file: str) -> fasttext.FastText._FastText:
    raise NotImplementedError("Implement train_model()")

# Save trained model to file
def save_model(model: fasttext.FastText._FastText, output_file: str) -> None:
    model.save_model(output_file)

# Load trained model from file
def load_model(input_file: str) -> fasttext.FastText._FastText:
    return fasttext.load_model(input_file)

# Test model on test dataset
def test_model(model: fasttext.FastText._FastText, test_file: str) -> Union[tuple, str]:
    raise NotImplementedError("Implement test_model()")

# Run the full pipeline
if __name__ == "__main__":
    save_dataset_in_fasttext_format()
    model = train_model('imdb.train')
    save_model(model, 'imdb.bin')
    test_results = test_model(model, 'imdb.test')
    print(test_results)

# Unit tests
class TestCases(unittest.TestCase):
    def test_get_label_description_valid(self):
        self.assertEqual(get_label_description(0), '__label__negative')
        self.assertEqual(get_label_description(1), '__label__positive')

    def test_get_label_description_invalid(self):
        with self.assertRaises(KeyError):
            get_label_description(2)
    
    def test_model(self):
        model = train_model('imdb.train')
        reviews = [
            {
                'text': 'A visually stunning sci-fi masterpiece that blends heart and high-concept storytelling with breathtaking precision. Every frame feels like a love letter to the genre.',
                'label': '__label__positive'
            },
            {
                'text': 'Sweet, quirky, and unexpectedly moving—this indie gem proves that even the smallest stories can leave the biggest impact. A delightful treat from start to finish."',
                'label': '__label__positive'
            },
            {
                'text': 'Space Ferrets: Galactic Mayhem is a cinematic black hole where plot, acting, and logic go to die. Watching it felt like being trapped in a never-ending Zoom call hosted by confused rodents.',
                'label': '__label__negative'
            }
        ]
        for review in reviews:
            self.assertEqual(
                model.predict([review['text']])[0],
                [[review['label']]]
            )

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
