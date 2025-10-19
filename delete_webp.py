#!/usr/bin/env python3
"""
Safe WebP deleter

By default this script does a dry-run and *moves* target WebP files to
<data_path>/.deleted_webp_<timestamp>/... so you can recover them if needed.

Use --permanent to permanently unlink (irreversible). Use --yes to skip
interactive confirmation.

Examples:
  # Dry-run (default)
  python delete_webp.py

  # Actually move files to a recovery folder
  python delete_webp.py --delete

  # Permanently remove (use with caution)
  python delete_webp.py --permanent --yes

This script operates on the same category list as convert_webp.py by default,
but you can pass a custom data path or set --recursive to search subfolders.
"""

from pathlib import Path
import argparse
import shutil
import time
import sys

CATEGORIES = ['Belts', 'Keyboard', 'Shoes', 'Watch']


def find_webp_files(base: Path, categories, recursive: bool):
    files = []
    if categories:
        for cat in categories:
            folder = base / cat
            if not folder.exists():
                continue
            if recursive:
                files.extend(list(folder.rglob('*.webp')))
            else:
                files.extend(list(folder.glob('*.webp')))
    else:
        # no categories provided -> scan base
        if recursive:
            files = list(base.rglob('*.webp'))
        else:
            files = list(base.glob('*.webp'))
    return files


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Safely delete/move WebP files in dataset folders')
    parser.add_argument('data_path', nargs='?', default=None,
                        help='Path to data folder (default: project/data)')
    parser.add_argument('--categories', '-c', nargs='+', default=None,
                        help='List of categories to operate on (default: use internal list)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Search subfolders recursively')
    parser.add_argument('--delete', action='store_true', help='Actually move files to recovery folder (default: dry-run)')
    parser.add_argument('--permanent', action='store_true', help='Permanently unlink files instead of moving them (must be combined with --delete)')
    parser.add_argument('--yes', '-y', action='store_true', help='Assume yes to confirmation prompts')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    base = Path(args.data_path) if args.data_path else project_root / 'data'

    cats = args.categories if args.categories is not None else CATEGORIES

    print('='*60)
    print('WebP Deleter')
    print('='*60)
    print(f'Data path: {base}')
    print('Categories:', ', '.join(cats) if cats else '(scan whole data folder)')
    print('Recursive:', args.recursive)
    print('Dry run (no files moved) by default. Use --delete to actually move files.')
    if args.permanent:
        print('WARNING: --permanent will permanently delete files (irreversible)')
    print()

    files = find_webp_files(base, cats, args.recursive)
    if not files:
        print('No .webp files found.')
        return

    print(f'Found {len(files)} .webp files:')
    for f in files[:200]:
        print('  ', f)
    if len(files) > 200:
        print('  ... (only first 200 shown)')

    if not args.delete:
        print('\nDry-run mode: no files will be moved or deleted.\n')
        return

    # confirm action
    if not args.yes:
        resp = input('Proceed with action (move files to recovery folder' + (', then permanently delete' if args.permanent else '') + ')? [y/N]: ')
        if resp.lower() not in ('y', 'yes'):
            print('Aborted by user.')
            return

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    recovery_root = base / f'.deleted_webp_{timestamp}'

    moved = 0
    failed = 0
    deleted = 0

    for f in files:
        try:
            rel = f.relative_to(base)
        except Exception:
            rel = Path(f.name)
        dest = recovery_root / rel
        ensure_dir(dest)
        try:
            if args.permanent:
                # remove permanently
                f.unlink()
                deleted += 1
            else:
                # move to recovery folder
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest))
                moved += 1
        except Exception as e:
            print(f'  ✗ Failed to process {f}: {e}')
            failed += 1

    print('\nSummary:')
    if args.permanent:
        print(f'  Permanently deleted: {deleted}')
    else:
        print(f'  Moved to recovery folder: {moved}')
        print(f'  Recovery folder: {recovery_root}')
    print(f'  Failed: {failed}')

    if not args.permanent and moved > 0:
        print('\nYou can restore files from the recovery folder if needed.')


if __name__ == '__main__':
    main()
