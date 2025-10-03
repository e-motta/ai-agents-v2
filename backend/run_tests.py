#!/usr/bin/env python3
"""
Test runner script for the FastAPI application.

This script provides convenient commands to run different test suites,
generate test reports, and coverage analysis.
"""

import argparse
import subprocess
import sys


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
    parser = argparse.ArgumentParser(
        description="Run tests for the FastAPI application"
    )
    parser.add_argument(
        "--type",
        choices=[
            "all",
            "unit",
            "router",
            "math",
            "chat",
            "coverage",
            "coverage-html",
            "coverage-report",
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

    args = parser.parse_args()

    # Base pytest command
    base_cmd = ["python3", "-m", "pytest"]

    if args.verbose:
        base_cmd.append("-v")

    if args.no_warnings:
        base_cmd.extend(["--disable-warnings"])

    # Coverage options
    coverage_args = ["--cov=app", "--cov-report=term"]
    if args.coverage_fail_under:
        coverage_args.append(f"--cov-fail-under={args.coverage_fail_under}")

    # Test selection
    test_commands = {
        "all": {
            "cmd": [
                *base_cmd,
                "tests/agents/test_router_agent.py",
                "tests/agents/test_math_agent.py",
                "tests/api/v1/test_chat_api.py",
            ],
            "description": "All Tests (Router + Math + Chat API)",
        },
        "unit": {
            "cmd": [
                *base_cmd,
                "tests/agents/test_router_agent.py",
                "tests/agents/test_math_agent.py",
            ],
            "description": "Unit Tests (Router + Math Agents)",
        },
        "router": {
            "cmd": [*base_cmd, "tests/agents/test_router_agent.py"],
            "description": "Router Agent Unit Tests",
        },
        "math": {
            "cmd": [*base_cmd, "tests/agents/test_math_agent.py"],
            "description": "Math Agent Unit Tests",
        },
        "chat": {
            "cmd": [*base_cmd, "tests/api/v1/test_chat_api.py"],
            "description": "Chat API E2E Tests",
        },
        "coverage": {
            "cmd": [
                *base_cmd,
                "tests/agents/test_router_agent.py",
                "tests/agents/test_math_agent.py",
                "tests/api/v1/test_chat_api.py",
                *coverage_args,
            ],
            "description": "Tests with Terminal Coverage Report",
        },
        "coverage-html": {
            "cmd": [
                *base_cmd,
                "tests/agents/test_router_agent.py",
                "tests/agents/test_math_agent.py",
                "tests/api/v1/test_chat_api.py",
                *coverage_args,
                "--cov-report=html",
            ],
            "description": "Tests with HTML Coverage Report",
        },
        "coverage-report": {
            "cmd": [
                *base_cmd,
                "tests/agents/test_router_agent.py",
                "tests/agents/test_math_agent.py",
                "tests/api/v1/test_chat_api.py",
                *coverage_args,
                "--cov-report=html",
                "--cov-report=xml",
            ],
            "description": "Tests with Comprehensive Coverage Reports (HTML + XML)",
        },
    }

    # Run the selected tests
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
