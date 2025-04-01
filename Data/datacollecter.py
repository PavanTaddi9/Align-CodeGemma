import ast
import json
import os
import tempfile
from git import Repo
from typing import Dict, List, Optional
import re

class FunctionWithTestsExtractor:
    def __init__(self):
        self.function_map: Dict[str, Dict] = {}
        self.current_module: str = ""
        self.jax_pattern = re.compile(r"jax\.|jnp\.|@jax\.jit")

    def is_jax_function(self, code: str) -> bool:
        return bool(self.jax_pattern.search(code))

    def parse_ast(self, file_path: str):
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.process_function(node, file_path)
            elif isinstance(node, ast.ClassDef) and "Test" in node.name:
                self.process_test_class(node, file_path)

    def process_function(self, node: ast.FunctionDef, file_path: str):
        func_code = ast.unparse(node)
        if not self.is_jax_function(func_code):
            return

        docstring = ast.get_docstring(node) or ""
        func_id = f"{self.current_module}.{node.name}"
        
        self.function_map[func_id] = {
            "function": func_code,
            "docstring": docstring.strip(),  # Clean whitespace
            "file": file_path,
            "tests": []
        }


    def process_test_class(self, node: ast.ClassDef, file_path: str):
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name.startswith("test"):
                self.process_test_function(item, file_path)

    def process_test_function(self, node: ast.FunctionDef, file_path: str):
        test_code = ast.unparse(node)
        assertions = [
            ast.unparse(n) for n in ast.walk(node)
            if isinstance(n, ast.Assert)
        ]
        
        # Find target functions through call patterns
        called_functions = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    called_functions.add(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    called_functions.add(n.func.attr)

        # Link tests to functions
        for func_id in self.function_map:
            func_name = func_id.split(".")[-1]
            if func_name in called_functions:
                self.function_map[func_id]["tests"].append({
                    "test_code": test_code,
                    "assertions": assertions,
                    "test_file": file_path
                })

class HybridTestCollector:
    def __init__(self):
        self.extractor = FunctionWithTestsExtractor()
    
    def process_repo(self, repo_url: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            clone_path = os.path.join(tmp_dir, repo_name)
            
            print(f"Cloning {repo_url}...")
            Repo.clone_from(repo_url, clone_path)
            
            for root, _, files in os.walk(clone_path):
                for file in files:
                    if file.endswith(".py"):
                        full_path = os.path.join(root, file)
                        self.extractor.current_module = self.get_module_name(full_path, clone_path)
                        self.extractor.parse_ast(full_path)
            
            return self.extractor.function_map

    def get_module_name(self, path: str, repo_root: str) -> str:
        rel_path = os.path.relpath(path, repo_root)
        return rel_path.replace("/", ".").replace(".py", "")

    def save_dataset(self, data: Dict, output_path: str):
        with open(output_path, "w") as f:
            json.dump(list(data.values()), f, indent=2)

# Usage
if __name__ == "__main__":
    collector = HybridTestCollector()
    
    # Example: JAX's official test suite
    dataset = collector.process_repo("https://github.com/jax-ml/jax")
    
    # Save with tests
    collector.save_dataset(dataset, "JAX_with_tests.json")