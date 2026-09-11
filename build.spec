# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for FSOC Coarse-Alignment Simulator.

Build a standalone executable:
    pyinstaller build.spec

The executable will be in dist/hexaverse/
"""

import os
import sys

block_cipher = None

# Collect all source files
src_path = os.path.join(os.path.dirname(os.path.abspath(SPEC)), 'src')

a = Analysis(
    [os.path.join(src_path, 'main.py')],
    pathex=[src_path],
    binaries=[],
    datas=[
        ('scenarios', 'scenarios'),
    ],
    hiddenimports=[
        'sim',
        'detect',
        'control',
        'config',
        'report',
        'evaluation',
        'yaml_config',
        'ai_detector',
        'generate_training_data',
        'train_detector',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'unittest',
        'test',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='hexaverse',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='hexaverse',
)
