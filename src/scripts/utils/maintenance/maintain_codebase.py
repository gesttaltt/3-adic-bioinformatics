import json
import os
import py_compile
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VSCODE_DIR = PROJECT_ROOT / ".vscode"
SETTINGS_FILE = VSCODE_DIR / "settings.json"
CSPELL_KEY = "cSpell.words"
VENV_DIR = ".venv"
IGNORED_DIRS = {VENV_DIR, ".git", "node_modules", "__pycache__"}


def _prune_ignored_dirs(dirs):
    """Remove ignored directories in-place to prevent os.walk traversal."""
    to_remove = [d for d in dirs if d in IGNORED_DIRS]
    for d in to_remove:
        dirs.remove(d)


def _parse_command(command):
    """Parse command string to list for secure subprocess execution."""
    if isinstance(command, list):
        return command
    if sys.platform == "win32":
        return command.split()
    return shlex.split(command)


def run_command(command, description):
    print(f"\nrunning {description}...")
    try:
        cmd_list = _parse_command(command)
        subprocess.run(cmd_list, check=True, shell=False, cwd=PROJECT_ROOT)
        print(f"✅ {description} complete.")
    except subprocess.CalledProcessError:
        print(f"❌ {description} failed or found issues.")


def run_formatter():
    run_command("black . --line-length 150", "Black (Formatter - Relaxed 150 chars)")
    run_command("isort .", "isort (Import Sorter)")


def run_linter_fixes():
    # Use Ruff for fast linting and auto-fixing
    print("\nrunning Ruff (Linter & Auto-Fixer)...")
    try:
        # We ignore errors (check=False) because ruff returns non-zero on found violations even if fixed
        cmd_list = _parse_command("ruff check --fix .")
        subprocess.run(cmd_list, check=False, shell=False, cwd=PROJECT_ROOT)
        print("✅ Ruff verification complete.")
    except Exception as e:
        print(f"⚠️ Ruff failed to run: {e}")


def add_words_to_dictionary(words):
    if not SETTINGS_FILE.exists():
        print(f"Creating {SETTINGS_FILE}")
        VSCODE_DIR.mkdir(exist_ok=True)
        settings = {CSPELL_KEY: []}
    else:
        try:
            with open(SETTINGS_FILE) as f:
                settings = json.load(f)
        except json.JSONDecodeError:
            settings = {CSPELL_KEY: []}

    current_words = set(settings.get(CSPELL_KEY, []))
    new_words_count = 0
    for w in words:
        if w not in current_words:
            current_words.add(w)
            new_words_count += 1
            print(f"Added '{w}' to dictionary.")

    if new_words_count > 0:
        settings[CSPELL_KEY] = sorted(list(current_words))
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"✅ Added {new_words_count} words to .vscode/settings.json")
    else:
        print("No new words to add.")


def run_mypy_check():
    print("\nrunning Mypy (Type Checker)...")
    try:
        # Run mypy on src directory
        cmd_list = _parse_command("mypy src")
        subprocess.run(
            cmd_list,
            check=False,  # Don't fail the whole script, just report
            shell=False,
            cwd=PROJECT_ROOT,
        )
        print("✅ Mypy check complete (see output above).")
    except Exception as e:
        print(f"⚠️ Mypy failed to run: {e}")


def run_syntax_check():
    print("\nrunning Syntax Check...")
    error_count = 0
    checked_count = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        _prune_ignored_dirs(dirs)

        for file in files:
            if file.endswith(".py"):
                path = str(Path(root) / file)
                checked_count += 1
                if not _check_syntax(path, file):
                    error_count += 1

    if error_count == 0:
        print(f"✅ Syntax Check Passed ({checked_count} file(s)).")
    else:
        print(f"❌ Found {error_count} syntax errors.")


def _check_syntax(path, filename):
    try:
        py_compile.compile(path, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ Syntax Error in {filename}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Could not compile {filename}: {e}")
        return False


def run_integrity_audit():
    print("\nrunning Integrity/Dependency Audit...")
    missing_files = []

    # Simple regex for finding file strings
    file_pattern = re.compile(r'["\']([^"\']+\.(pt|json|csv|pdb|fasta))["\']')

    for root, dirs, files in os.walk(PROJECT_ROOT):
        _prune_ignored_dirs(dirs)

        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                _check_file_integrity(path, file_pattern, missing_files)

    if not missing_files:
        print("✅ No missing data dependencies found.")
    else:
        print(f"⚠️ Found {len(missing_files)} missing data references (check 'integrity_report.log' for details).")


def _check_file_integrity(path, pattern, missing_files_list):
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
            matches = pattern.findall(content)
            for match in matches:
                filename = match[0]
                if "*" in filename or "{" in filename:
                    continue

                if not _dependency_exists(path, filename):
                    # Only report if it looks like a local file
                    if not filename.startswith("http"):
                        missing_files_list.append((str(path.relative_to(PROJECT_ROOT)), filename))
    except Exception:
        pass


def _dependency_exists(source_path, filename):
    # Check relative to script
    if (source_path.parent / filename).exists():
        return True
    # Check relative to project root
    return bool((PROJECT_ROOT / filename).exists())


def clean_artifacts():
    print("\nrunning Cleanup...")
    # Remove the temporary vocabulary file
    vocab_file = PROJECT_ROOT / "detected_unknowns.txt"
    if vocab_file.exists():
        try:
            os.remove(vocab_file)
            print(f"✅ Removed {vocab_file.name}")
        except Exception as e:
            print(f"❌ Failed to remove {vocab_file.name}: {e}")
    else:
        print("✅ No artifacts to clean.")


def run_tests():
    print("\nrunning Unit Tests...")
    try:
        # Run fast tests only, quiet mode
        cmd_list = _parse_command("pytest tests/unit --maxfail=5 -q")
        subprocess.run(
            cmd_list,
            check=True,
            shell=False,
            cwd=PROJECT_ROOT,
        )
        print("✅ Unit Tests Passed.")
    except subprocess.CalledProcessError:
        print("❌ Unit Tests Failed.")


def main():
    print("=== Codebase Maintenance Script ===")

    # 1. Format Code
    run_formatter()

    # 2. Syntax Check (New)
    run_syntax_check()

    # 3. Integrity Audit (New)
    run_integrity_audit()

    # 4. Unit Tests (New)
    run_tests()

    # 5. Type Check (New)
    run_mypy_check()

    # 2. Add Common Terms to Dictionary
    print("\nUpdating Dictionary...")
    BIO_TERMS = []
    # Check for auto-generated vocabulary list
    auto_vocab_file = PROJECT_ROOT / "detected_unknowns.txt"
    if auto_vocab_file.exists():
        print(f"Found auto-generated vocabulary: {auto_vocab_file}")
        try:
            with open(auto_vocab_file, encoding="utf-8") as f:
                auto_terms = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(auto_terms)} terms from scan.")
            BIO_TERMS.extend(auto_terms)
        except Exception as e:
            print(f"Error reading vocabulary file: {e}")

    add_words_to_dictionary(BIO_TERMS)

    # 5. Cleanup
    clean_artifacts()

    # 6. Auto-Fix Linter Issues (New)
    run_linter_fixes()

    print("\n=== Maintenance Complete ===")


if __name__ == "__main__":
    main()
