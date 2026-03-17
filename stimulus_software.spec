# stimulus_software.spec
# Build with: pyinstaller stimulus_software.spec
#
# macOS:  produces  dist/Stimulus Software.app  (zip for distribution)
# Linux:  produces  dist/stimulus_software       (single-file binary)

import platform

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('audio_data', 'audio_data'),
        ('lib/assets', 'lib/assets'),
    ],
    hiddenimports=[
        'scipy.signal',
        'scipy.signal._upfirdn',
        'scipy.signal._upfirdn_apply',
        'scipy._lib.messagestream',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if platform.system() == 'Darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='Stimulus Software',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='Stimulus Software',
    )
    app = BUNDLE(
        coll,
        name='Stimulus Software.app',
        icon=None,
        bundle_identifier='edu.eeg.stimulus-software',
        info_plist={
            'NSHighResolutionCapable': True,
            'LSBackgroundOnly': False,
            'CFBundleShortVersionString': '1.0.0',
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='stimulus_software',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
