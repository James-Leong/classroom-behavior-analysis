#!/bin/sh -e
set -x

# Usage: ./scripts/lint.sh [--all|--changed]
# - --all (default): lint/format the whole `app` package
# - --changed: lint/format only Python files changed in the working tree (staged/modified)
# Note: positional 'changed' or 'all' are NOT supported; use `-c/--changed` or `-a/--all`.

mode="all"

print_usage() {
	printf "Usage: %s [--all|--changed]\n" "$0"
	printf "  --all      Lint/format the whole app (default)\n"
	printf "  --changed  Lint/format only changed Python files (staged/modified)\n"
	printf "  -a/--all and -c/--changed are supported. Positional 'all'/'changed' are not.\n"
}

# simple arg parsing: accept only -c/--changed or -a/--all (no positional 'changed'/'all')
mode_explicit=0
files_list=""
if [ $# -gt 0 ]; then
	for arg in "$@"; do
		case "$arg" in
			-c|--changed)
				mode="changed"
				mode_explicit=1
				;;
			-a|--all)
				mode="all"
				mode_explicit=1
				;;
			-h|--help)
				print_usage
				exit 0
				;;
			--*)
				echo "Unknown option: $arg"
				print_usage
				exit 2
				;;
			*)
				# collect non-option args as files (pre-commit will pass filenames)
				files_list="${files_list}\n${arg}"
				;;
		esac
	done
fi

# If file arguments were provided and no explicit mode flag, lint those files and exit
if [ -n "$(printf "%b" "$files_list" | sed '/^$/d')" ] && [ "$mode_explicit" -eq 0 ]; then
	files=$(printf "%b" "$files_list" | sed '/^$/d' | sort -u)
	printf "%s\n" "$files" | xargs -r ruff check
	printf "%s\n" "$files" | xargs -r ruff format
	exit 0
fi

if [ "$mode" = "changed" ]; then
	staged=$(git diff --name-only --diff-filter=ACMRTUXB --cached 2>/dev/null | grep -E '\.py$' || true)
	modified=$(git ls-files -m 2>/dev/null | grep -E '\.py$' || true)

	files=$(printf "%s\n%s\n" "$staged" "$modified" | sort -u | sed '/^$/d')

	if [ -z "$files" ]; then
		echo "No changed Python files found. Nothing to lint/format."
		exit 0
	fi

	printf "%s\n" "$files" | xargs -r ruff check
	printf "%s\n" "$files" | xargs -r ruff format
else
	ruff check app
	ruff format app
fi