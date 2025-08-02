#!/usr/bin/env python3
"""
Comprehensive dependency fixer and requirements updater for UPHire v2
This script helps resolve package conflicts and updates requirements.txt
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pkg_resources
import importlib.util


def get_installed_packages() -> Dict[str, str]:
    """Get all currently installed packages with their versions."""
    installed_packages = {}
    for dist in pkg_resources.working_set:
        installed_packages[dist.project_name.lower()] = dist.version
    return installed_packages


def get_package_dependencies(package_name: str) -> Set[str]:
    """Get dependencies for a specific package."""
    try:
        dist = pkg_resources.get_distribution(package_name)
        deps = set()
        for req in dist.requires():
            deps.add(req.project_name.lower())
        return deps
    except:
        return set()


def check_import_availability(module_name: str) -> bool:
    """Check if a module can be imported successfully."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ImportError:
        return False


def analyze_project_imports() -> Set[str]:
    """Analyze Python files to find required packages."""
    project_root = Path(".")
    required_packages = set()

    # Common import mappings
    import_to_package = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "starlette": "starlette",
        "pydantic": "pydantic",
        "pymongo": "pymongo",
        "motor": "motor",
        "numpy": "numpy",
        "torch": "torch",
        "transformers": "transformers",
        "sentence_transformers": "sentence-transformers",
        "huggingface_hub": "huggingface_hub",
        "langchain": "langchain",
        "openai": "openai",
        "groq": "groq",
        "PyPDF2": "pypdf2",
        "docx": "python-docx",
        "requests": "requests",
        "httpx": "httpx",
        "aiofiles": "aiofiles",
        "psutil": "psutil",
        "cryptography": "cryptography",
        "jwt": "pyjwt",
        "jose": "python-jose",
        "passlib": "passlib",
        "bcrypt": "bcrypt",
        "email_validator": "email-validator",
        "spacy": "spacy",
        "nltk": "nltk",
        "pandas": "pandas",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "plotly": "plotly",
        "seaborn": "seaborn",
        "tqdm": "tqdm",
        "rich": "rich",
        "structlog": "structlog",
        "orjson": "orjson",
        "msgpack": "msgpack",
        "regex": "regex",
        "dateutil": "python-dateutil",
        "dotenv": "python-dotenv",
        "multipart": "python-multipart",
    }

    # Scan Python files for imports
    for py_file in project_root.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Find import statements
            import_lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip().startswith(("import ", "from "))
            ]

            for line in import_lines:
                # Extract module name
                if line.startswith("import "):
                    module = line.replace("import ", "").split()[0].split(".")[0]
                elif line.startswith("from "):
                    module = line.replace("from ", "").split()[0].split(".")[0]
                else:
                    continue

                # Map to package name
                if module in import_to_package:
                    required_packages.add(import_to_package[module])
                elif not module.startswith(
                    (
                        "apis",
                        "core",
                        "schemas",
                        "embeddings",
                        "GroqcloudLLM",
                        "Rag",
                        "mangodatabase",
                        "multipleresumepraser",
                        "masking",
                        "apisofmango",
                        "Expericecal",
                        "Retrivers",
                        "recent_search_uts",
                        "textextractors",
                        "properties",
                    )
                ):
                    # Add unknown external packages
                    required_packages.add(module.lower())

        except Exception as e:
            print(f"Warning: Could not analyze {py_file}: {e}")
            continue

    return required_packages


def check_version_conflicts() -> Dict[str, List[str]]:
    """Check for known version conflicts."""
    conflicts = {}
    installed = get_installed_packages()

    # Known problematic combinations
    conflict_rules = [
        {
            "packages": ["transformers", "huggingface_hub"],
            "rule": "transformers >= 4.42.0 requires huggingface_hub >= 0.25.0",
            "fix": "Update both packages to compatible versions",
        },
        {
            "packages": ["torch", "transformers"],
            "rule": "torch 2.6.0+ may conflict with transformers < 4.45.0",
            "fix": "Update transformers or downgrade torch",
        },
        {
            "packages": ["pydantic", "fastapi"],
            "rule": "pydantic 2.x requires fastapi >= 0.100.0",
            "fix": "Ensure FastAPI is updated for Pydantic v2",
        },
        {
            "packages": ["langchain", "langchain-core"],
            "rule": "LangChain components must have compatible versions",
            "fix": "Update all langchain-* packages together",
        },
    ]

    for rule in conflict_rules:
        packages_present = [p for p in rule["packages"] if p in installed]
        if len(packages_present) > 1:
            versions = [f"{p}=={installed[p]}" for p in packages_present]
            conflicts[rule["rule"]] = {"packages": versions, "fix": rule["fix"]}

    return conflicts


def create_updated_requirements():
    """Create updated requirements.txt with proper version ranges."""

    installed = get_installed_packages()
    project_imports = analyze_project_imports()

    # Core requirements with version ranges
    core_requirements = {
        # Web Framework
        "fastapi": ">=0.100.0,<0.120.0",
        "uvicorn": ">=0.23.0,<0.36.0",
        "starlette": ">=0.40.0,<0.48.0",
        "gunicorn": ">=21.0.0,<24.0.0",
        # Database
        "pymongo": ">=4.13.0,<5.0.0",
        "motor": ">=3.3.0,<4.0.0",
        # ML/AI
        "torch": ">=2.1.0,<3.0.0",
        "transformers": ">=4.42.0,<4.52.0",
        "sentence-transformers": ">=2.2.2,<5.0.0",
        "huggingface_hub": ">=0.25.0,<1.0.0",
        "tokenizers": ">=0.19.0,<1.0.0",
        "safetensors": ">=0.4.0,<1.0.0",
        "numpy": ">=1.24.0,<2.0.0",
        "scikit-learn": ">=1.3.0,<2.0.0",
        "scipy": ">=1.10.0,<2.0.0",
        # LangChain
        "langchain": ">=0.3.20,<0.4.0",
        "langchain-community": ">=0.3.20,<0.4.0",
        "langchain-core": ">=0.3.60,<0.4.0",
        "langchain-openai": ">=0.3.20,<1.0.0",
        "langchain-groq": ">=0.3.0,<1.0.0",
        "langchain-huggingface": ">=0.3.0,<1.0.0",
        "langchain-mongodb": ">=0.6.0,<1.0.0",
        "langchain-ollama": ">=0.3.0,<1.0.0",
        "langchain-google-genai": ">=2.1.0,<3.0.0",
        "langchain-text-splitters": ">=0.3.0,<1.0.0",
        "langgraph": ">=0.4.0,<1.0.0",
        "langsmith": ">=0.4.0,<1.0.0",
        # LLM APIs
        "openai": ">=1.80.0,<2.0.0",
        "groq": ">=0.28.0,<1.0.0",
        "google-genai": ">=1.16.0,<2.0.0",
        "ollama": ">=0.5.0,<1.0.0",
        # Data Processing
        "pydantic": ">=2.6.0,<3.0.0",
        "pydantic-settings": ">=2.0.0,<3.0.0",
        "pandas": ">=2.0.0,<3.0.0",
        "openpyxl": ">=3.1.0,<4.0.0",
        # Document Processing
        "pypdf2": ">=3.0.0,<4.0.0",
        "python-docx": ">=1.0.0,<2.0.0",
        "pdfplumber": ">=0.10.0,<1.0.0",
        # Security
        "cryptography": ">=45.0.0,<46.0.0",
        "pyopenssl": ">=25.0.0,<26.0.0",
        "bcrypt": ">=4.0.0,<5.0.0",
        "passlib": ">=1.7.0,<2.0.0",
        "python-jose": ">=3.3.0,<4.0.0",
        "pyjwt": ">=2.8.0,<3.0.0",
        # HTTP/Async
        "requests": ">=2.31.0,<3.0.0",
        "httpx": ">=0.24.0,<1.0.0",
        "aiohttp": ">=3.9.0,<4.0.0",
        "aiofiles": ">=23.0.0,<25.0.0",
        # Utilities
        "python-dateutil": ">=2.8.0,<3.0.0",
        "python-dotenv": ">=1.0.0,<2.0.0",
        "python-multipart": ">=0.0.6,<1.0.0",
        "typing-extensions": ">=4.12.0,<5.0.0",
        "email-validator": ">=2.0.0,<3.0.0",
        "psutil": ">=5.9.0,<8.0.0",
        # Text Processing
        "tqdm": ">=4.65.0,<5.0.0",
        "regex": ">=2024.0.0",
        "rich": ">=13.0.0,<14.0.0",
        "structlog": ">=23.0.0,<25.0.0",
        # Optional but useful
        "orjson": ">=3.9.0,<4.0.0",
        "python-ulid": ">=3.0.0,<4.0.0",
    }

    # Filter requirements based on what's actually used
    filtered_requirements = {}
    for package, version in core_requirements.items():
        package_key = package.replace("-", "_").lower()
        if (
            package_key in project_imports
            or package.lower() in project_imports
            or package in installed
        ):
            filtered_requirements[package] = version

    return filtered_requirements


def generate_fix_script():
    """Generate a comprehensive fix script."""
    fix_commands = [
        "# Comprehensive dependency fix script",
        "# Run this script to resolve dependency conflicts",
        "",
        "echo 'Starting dependency fix process...'",
        "",
        "# Step 1: Backup current environment",
        "pip freeze > requirements_backup.txt",
        "echo 'Current environment backed up to requirements_backup.txt'",
        "",
        "# Step 2: Uninstall problematic packages",
        "echo 'Removing potentially conflicting packages...'",
        "pip uninstall -y transformers huggingface_hub tokenizers safetensors sentence-transformers",
        "pip uninstall -y torch torchvision torchaudio",
        "pip uninstall -y langchain langchain-community langchain-core langchain-openai langchain-groq",
        "",
        "# Step 3: Clear pip cache",
        "pip cache purge",
        "",
        "# Step 4: Install core dependencies in correct order",
        "echo 'Installing core dependencies...'",
        "pip install --upgrade pip setuptools wheel",
        "",
        "# Install PyTorch first (CPU version for better compatibility)",
        "pip install 'torch>=2.1.0,<3.0.0' --index-url https://download.pytorch.org/whl/cpu",
        "",
        "# Install HuggingFace ecosystem",
        "pip install 'huggingface_hub>=0.25.0,<1.0.0'",
        "pip install 'tokenizers>=0.19.0,<1.0.0'",
        "pip install 'safetensors>=0.4.0,<1.0.0'",
        "pip install 'transformers>=4.42.0,<4.52.0'",
        "pip install 'sentence-transformers>=2.2.2,<5.0.0'",
        "",
        "# Install FastAPI ecosystem",
        "pip install 'fastapi>=0.100.0,<0.120.0'",
        "pip install 'uvicorn>=0.23.0,<0.36.0'",
        "pip install 'pydantic>=2.6.0,<3.0.0'",
        "",
        "# Install remaining requirements",
        "pip install -r requirements.txt",
        "",
        "# Step 5: Verify installation",
        "echo 'Verifying installation...'",
        "python -c \"import transformers, sentence_transformers, fastapi, torch; print('All core packages imported successfully!')\"",
        "",
        "echo 'Dependency fix completed!'",
    ]

    return "\n".join(fix_commands)


def main():
    """Main function to analyze and fix dependencies."""
    print("🔍 Analyzing UPHire v2 dependencies...")

    # Get current state
    installed = get_installed_packages()
    project_imports = analyze_project_imports()
    conflicts = check_version_conflicts()

    print(f"\n📦 Found {len(installed)} installed packages")
    print(f"🔧 Detected {len(project_imports)} required packages from code analysis")

    if conflicts:
        print(f"\n⚠️  Found {len(conflicts)} potential conflicts:")
        for rule, details in conflicts.items():
            print(f"  - {rule}")
            print(f"    Packages: {', '.join(details['packages'])}")
            print(f"    Fix: {details['fix']}")

    # Generate updated requirements
    print("\n📝 Generating updated requirements.txt...")
    updated_reqs = create_updated_requirements()

    # Save updated requirements
    req_lines = []
    req_lines.append("# Updated requirements.txt for UPHire v2")
    req_lines.append("# Generated by dependency analyzer")
    req_lines.append("")

    categories = {
        "# Web Framework": ["fastapi", "uvicorn", "starlette", "gunicorn"],
        "# Database": ["pymongo", "motor"],
        "# ML/AI Core": ["torch", "numpy", "scikit-learn", "scipy"],
        "# NLP/Transformers": [
            "transformers",
            "sentence-transformers",
            "huggingface_hub",
            "tokenizers",
            "safetensors",
        ],
        "# LangChain": [
            p
            for p in updated_reqs.keys()
            if p.startswith("langchain") or p in ["langsmith", "langgraph"]
        ],
        "# LLM APIs": ["openai", "groq", "google-genai", "ollama"],
        "# Data Processing": ["pydantic", "pydantic-settings", "pandas", "openpyxl"],
        "# Document Processing": ["pypdf2", "python-docx", "pdfplumber"],
        "# Security": [
            "cryptography",
            "pyopenssl",
            "bcrypt",
            "passlib",
            "python-jose",
            "pyjwt",
        ],
        "# HTTP/Async": ["requests", "httpx", "aiohttp", "aiofiles"],
        "# Utilities": [
            "python-dateutil",
            "python-dotenv",
            "python-multipart",
            "typing-extensions",
            "email-validator",
            "psutil",
            "tqdm",
            "regex",
            "rich",
            "structlog",
            "orjson",
            "python-ulid",
        ],
    }

    for category, packages in categories.items():
        req_lines.append(category)
        for package in packages:
            if package in updated_reqs:
                req_lines.append(f"{package}{updated_reqs[package]}")
        req_lines.append("")

    # Write updated requirements
    with open("requirements_updated.txt", "w") as f:
        f.write("\n".join(req_lines))

    # Generate fix script
    fix_script = generate_fix_script()
    with open("fix_dependencies.sh", "w") as f:
        f.write(fix_script)

    # Windows batch version
    win_fix = fix_script.replace("echo '", "echo ").replace("'", "")
    with open("fix_dependencies.bat", "w") as f:
        f.write(win_fix)

    print("\n✅ Analysis complete!")
    print("📄 Files generated:")
    print("  - requirements_updated.txt (comprehensive requirements)")
    print("  - fix_dependencies.sh (Unix fix script)")
    print("  - fix_dependencies.bat (Windows fix script)")

    print("\n🚀 Next steps:")
    print("1. Backup your current environment: pip freeze > backup.txt")
    print("2. Run the fix script: ./fix_dependencies.sh (or .bat on Windows)")
    print("3. Or manually install: pip install -r requirements_updated.txt")

    if conflicts:
        print("\n⚠️  Address these conflicts first:")
        for rule, details in conflicts.items():
            print(f"  - {details['fix']}")


if __name__ == "__main__":
    main()
