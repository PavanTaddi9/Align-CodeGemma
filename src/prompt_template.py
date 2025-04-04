import json
import pandas as pd
from datasets import Dataset

# Define the system prompt (for structure)
SYSTEM_PROMPT = """You are an expert AI assistant specializing in generating highly efficient, well-structured, and optimized code.  
Follow these principles:
1. **Prioritize efficiency** Use the most optimal algorithms and minimize computational overhead.
2. **Leverage standard libraries** Use built-in or well-optimized library functions instead of unnecessary custom implementations.
3. **Ensure correctness and robustness** Handle edge cases, numerical stability, and performance bottlenecks.
4. **Provide clear, modular, and well-documented code** Make the code readable, maintainable, and scalable.
5. **Verify with test cases** Include Assert test cases to confirm correctness.
"""

# Function to format each instruction into the structured prompt
def format_instruction(instruction):
    return f"""
<system>
{SYSTEM_PROMPT}
</system>

<instruction>
{instruction}
</instruction>

<response>
<think>
Analyze the problem carefully. Think step by step, Identify efficient solutions using appropriate data structures and optimized library functions.
</think>

<code>
# Your optimized code implementation goes here.
</code>

<test>
# Test cases with assert statements to check the functionality of genearted code.
</test>
</response>
"""

