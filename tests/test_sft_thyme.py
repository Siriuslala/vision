"""Geometry, chronology, and source-code rejection checks for Thyme conversion."""
import base64
import io
import unittest

from PIL import Image

from data.sft.datasets.thyme import CropParser, convert_row


def encoded(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class ThymeTest(unittest.TestCase):
    def test_nested_crop_uses_original_coordinates_and_reference(self):
        parser = CropParser("original.jpg", (100, 50))
        first = parser.parse('p="original.jpg"\na=Image.open(p)\nb=a.crop((10,10,90,40))\nb.save("out.jpg")')
        second = parser.parse('c=b.resize((160,60))\nd=c.crop((20,10,100,50))\nd.save("out2.jpg")')
        self.assertEqual(first.box, (10, 10, 90, 40))
        self.assertEqual(second.box, (20, 15, 60, 35))
        self.assertEqual(second.reference, 1)

    def test_arbitrary_execution_and_augmentation_rejected(self):
        for code in ['import subprocess\nsubprocess.run("x")', 'x=__import__("os")',
                     'a=Image.open("other.jpg")', 'a=Image.open("original.jpg")\nb=a.rotate(90)',
                     'a=Image.open("original.jpg")\nb=a.crop((-1,0,10,10))']:
            with self.subTest(code=code), self.assertRaises(ValueError):
                CropParser("original.jpg", (100, 50)).parse(code)

    def test_true_interleaving_and_answer_boundary(self):
        original = Image.new("RGB", (100, 50), "red")
        row = {"image": [encoded(original), encoded(original.crop((10, 5, 60, 30)))],
               "question": '<image>\nWhat color is the object?\n### User Image Path:** "/data/gqa/images/1.jpg"\n### User Image Size:** "100x50"',
               "response": '<think>Inspect the object.\n<code>\n```python\np="/data/gqa/images/1.jpg"\na=Image.open(p)\nb=a.crop((10,5,60,30))\nb.save("out.jpg")\n```\n</code><sandbox_output><image></sandbox_output>It is red.</think><answer>red</answer>'}
        result = convert_row(row, "test:1", "revision", "2round")
        self.assertEqual([x["type"] for x in result["events"]], ["reasoning_text", "visual_read", "reasoning_text"])
        self.assertEqual(result["events"][1]["box"], [10, 5, 60, 30])
        self.assertEqual(result["query"], "What color is the object?")
        self.assertEqual(result["answer"], {"type": "text", "value": "red"})
        self.assertFalse(result["visual_only_safe"])
        row["image"][1] = encoded(Image.new("RGB", (50, 25), "blue"))
        with self.assertRaisesRegex(ValueError, "derived image mismatch"):
            convert_row(row, "test:1", "revision", "2round")


if __name__ == "__main__":
    unittest.main()
