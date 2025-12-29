"""
Example: Pretrain a small GPT model on Shakespeare text.

This example demonstrates how to pretrain a GPT model from scratch.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Example Shakespeare text (sample)
shakespeare_sample = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely; but
they think we are too dear: the leanness that afflicts
us, the object of our misery, is as an inventory to
particularise their abundance; our sufferance is a gain
to them Let us revenge this with our pikes, ere we
become rakes: for the gods know I speak this in hunger
for bread, not in thirst for revenge.
"""

# Create data directory
os.makedirs('data', exist_ok=True)

# Save sample data
data_path = 'data/shakespeare_sample.txt'
with open(data_path, 'w', encoding='utf-8') as f:
    f.write(shakespeare_sample * 10)  # Repeat to have more data

print(f"Sample data saved to {data_path}")
print("\nTo pretrain the model, run:")
print(f"python pretrain.py --data_path {data_path} --output_dir outputs/shakespeare --num_epochs 5 --batch_size 16")
