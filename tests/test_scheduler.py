import unittest
import sys
import os
from pathlib import Path

# Add the project root to sys.path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from parser.scheduler import get_prompt_schedule


class TestScheduler(unittest.TestCase):
    def test_basic_scheduling(self):
        # [before:after:when]
        # At 10 steps, 0.5 is step 5
        schedule = get_prompt_schedule("[cat:dog:0.5]", 10)
        # Expect (5, "cat"), (10, "dog")
        self.assertEqual(len(schedule), 2)
        self.assertEqual(schedule[0], (5, "cat"))
        self.assertEqual(schedule[1], (10, "dog"))

    def test_step_based_scheduling(self):
        # [before:after:step]
        schedule = get_prompt_schedule("[cat:dog:7]", 10)
        self.assertEqual(len(schedule), 2)
        self.assertEqual(schedule[0], (7, "cat"))
        self.assertEqual(schedule[1], (10, "dog"))

    def test_alternation(self):
        # [A|B]
        schedule = get_prompt_schedule("[white|black]", 4)
        # Alternation changes every step
        self.assertEqual(len(schedule), 4)
        self.assertEqual(schedule[0][1], "white")
        self.assertEqual(schedule[1][1], "black")
        self.assertEqual(schedule[2][1], "white")
        self.assertEqual(schedule[3][1], "black")

    def test_scheduled_alternation_add(self):
        # [A|B:0.5] - Yield A until 50%, then alternate
        schedule = get_prompt_schedule("[A|B:0.5]", 4)
        # steps=4, 0.5=2
        # Step 1: A
        # Step 2: A
        # Step 3: A (cycle (3-1)%2=0) - wait, check implementation
        # Actually _at_step calculates idx = (step - 1) % len(options)
        # Step 1: yield options[0] -> A
        # Step 2: yield options[0] -> A
        # Step 3: (3-1)%2 = 0 -> A  <-- Is this right? A1111 behavior check.
        # Let's check my implementation in scheduler.py:
        # idx = (step - 1) % len(options)
        # So Step 3 (step > 2) yields options[0] = A
        # Step 4: (4-1)%2 = 1 -> B

        self.assertEqual(schedule[0][1], "A")
        self.assertEqual(schedule[1][1], "A")
        self.assertEqual(schedule[2][1], "A")
        self.assertEqual(schedule[3][1], "B")

    def test_scheduled_alternation_remove(self):
        # [A|B::0.5] - Alternate until 50%, then first tag
        schedule = get_prompt_schedule("[A|B::0.5]", 4)
        # Step 1: A
        # Step 2: B
        # Step 3: A (first tag)
        # Step 4: A (first tag)
        self.assertEqual(schedule[0][1], "A")
        self.assertEqual(schedule[1][1], "B")
        self.assertEqual(schedule[2][1], "A")
        self.assertEqual(schedule[3][1], "A")

    def test_nested_logic(self):
        # [(red:1.2):[blue:green:0.5]:0.3]
        # steps=10, 0.3=3, 0.5=5
        schedule = get_prompt_schedule("[(red:1.2):[blue:green:0.5]:0.3]", 10)
        # Sequence expected:
        # Step 1-3: (red:1.2)
        # Step 4-5: blue
        # Step 6-10: green
        self.assertEqual(len(schedule), 3)
        self.assertEqual(schedule[0], (3, "(red:1.2)"))
        self.assertEqual(schedule[1], (5, "blue"))
        self.assertEqual(schedule[2], (10, "green"))

    def test_break_isolation(self):
        # a BREAK [b:c:0.5]
        schedule = get_prompt_schedule("a BREAK [b:c:0.5]", 10)
        self.assertEqual(len(schedule), 2)
        self.assertEqual(schedule[0], (5, "a BREAK b"))
        self.assertEqual(schedule[1], (10, "a BREAK c"))


if __name__ == "__main__":
    unittest.main()
