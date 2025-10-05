#!/usr/bin/env python3
"""
Test runner script for the FastAPI application.

This script provides convenient commands to run different test suites,
generate test reports, and coverage analysis.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def discover_test_files() -> dict[str, list[str]]:
    """
    Dynamically discover all test files in the tests directory.

    Returns:
        Dictionary mapping test categories to lists of test file paths
    """
    tests_dir = Path("tests")
    if not tests_dir.exists():
        return {}

    test_categories = {}

    # Walk through all subdirectories in tests/
    for test_file in tests_dir.rglob("test_*.py"):
        # Get the relative path from tests/
        relative_path = test_file.relative_to(tests_dir)

        # Determine category based on directory structure
        category = "root" if len(relative_path.parts) == 1 else relative_path.parts[0]

        # Convert to string path for pytest
        test_path = str(test_file)

        if category not in test_categories:
            test_categories[category] = []
        test_categories[category].append(test_path)

    return test_categories


def get_all_test_files() -> list[str]:
    """Get all test files for running complete test suite."""
    test_categories = discover_test_files()
    all_tests = []

    for category_tests in test_categories.values():
        all_tests.extend(category_tests)

    return sorted(all_tests)


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"{'=' * 60}")

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner function."""
    # Discover available test categories dynamically
    test_categories = discover_test_files()
    available_categories = list(test_categories.keys())

    parser = argparse.ArgumentParser(
        description="Run tests for the FastAPI application"
    )
    parser.add_argument(
        "--type",
        choices=[
            "all",
            "unit",
            "coverage",
            "coverage-html",
            "coverage-report",
            *available_categories,
        ],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests in verbose mode"
    )
    parser.add_argument("--no-warnings", action="store_true", help="Suppress warnings")
    parser.add_argument(
        "--coverage-threshold",
        type=int,
        default=80,
        help="Minimum coverage percentage threshold (default: 80)",
    )
    parser.add_argument(
        "--coverage-fail-under",
        type=int,
        help="Exit with non-zero status if coverage is below this percentage",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available test categories and exit",
    )

    args = parser.parse_args()

    # List available categories if requested
    if args.list_categories:
        print("Available test categories:")
        print(f"  all - Run all tests ({len(get_all_test_files())} files)")
        print("  unit - Run unit tests (agents + core)")
        print("  coverage - Run tests with coverage report")
        print("  coverage-html - Run tests with HTML coverage report")
        print("  coverage-report - Run tests with comprehensive coverage reports")
        print("\nCategory-specific tests:")
        for category, files in test_categories.items():
            print(f"  {category} - {len(files)} test file(s)")
            for file in sorted(files):
                print(f"    • {file}")
        return

    # Base pytest command - use python from current environment
    base_cmd = ["python", "-m", "pytest"]

    if args.verbose:
        base_cmd.append("-v")

    if args.no_warnings:
        base_cmd.extend(["--disable-warnings"])

    # Coverage options
    coverage_args = ["--cov=app", "--cov-report=term"]
    if args.coverage_fail_under:
        coverage_args.append(f"--cov-fail-under={args.coverage_fail_under}")

    # Build test commands dynamically
    test_commands = {}

    # All tests
    all_test_files = get_all_test_files()
    test_commands["all"] = {
        "cmd": [*base_cmd, *all_test_files],
        "description": f"All Tests ({len(all_test_files)} files)",
    }

    # Unit tests (agents + core)
    unit_tests = []
    for category in ["agents", "core"]:
        if category in test_categories:
            unit_tests.extend(test_categories[category])
    test_commands["unit"] = {
        "cmd": [*base_cmd, *unit_tests],
        "description": f"Unit Tests ({len(unit_tests)} files)",
    }

    # Coverage variants
    test_commands["coverage"] = {
        "cmd": [*base_cmd, *all_test_files, *coverage_args],
        "description": (
            f"Tests with Terminal Coverage Report ({len(all_test_files)} files)"
        ),
    }
    test_commands["coverage-html"] = {
        "cmd": [*base_cmd, *all_test_files, *coverage_args, "--cov-report=html"],
        "description": f"Tests with HTML Coverage Report ({len(all_test_files)} files)",
    }
    test_commands["coverage-report"] = {
        "cmd": [
            *base_cmd,
            *all_test_files,
            *coverage_args,
            "--cov-report=html",
            "--cov-report=xml",
        ],
        "description": (
            f"Tests with Comprehensive Coverage Reports ({len(all_test_files)} files)"
        ),
    }

    # Category-specific tests
    for category, files in test_categories.items():
        test_commands[category] = {
            "cmd": [*base_cmd, *files],
            "description": f"{category.title()} Tests ({len(files)} files)",
        }

    # Run the selected tests
    if args.type not in test_commands:
        print(f"❌ Unknown test type: {args.type}")
        print("Use --list-categories to see available options")
        sys.exit(1)

    test_config = test_commands[args.type]
    success = run_command(test_config["cmd"], test_config["description"])

    if success:
        print("\n🎉 All tests passed successfully!")

        # Show coverage report information
        if "coverage" in args.type:
            print("\n📊 Coverage Reports Generated:")
            if args.type == "coverage":
                print("   • Terminal report displayed above")
            elif args.type == "coverage-html":
                print("   • Terminal report displayed above")
                print("   • HTML report: htmlcov/index.html")
            elif args.type == "coverage-report":
                print("   • Terminal report displayed above")
                print("   • HTML report: htmlcov/index.html")
                print("   • XML report: coverage.xml")

            print(f"\n💡 Coverage Threshold: {args.coverage_threshold}%")
            if args.coverage_fail_under:
                print(f"💡 Fail Under Threshold: {args.coverage_fail_under}%")
    else:
        print("\n💥 Some tests failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
