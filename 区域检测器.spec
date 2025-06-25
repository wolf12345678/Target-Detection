# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Main2.py'],
    pathex=[],
    binaries=[],
    datas=[('encrypted_model.bin', '.')],
    hiddenimports=['torch', 'torchvision', 'cryptography', 'cryptography.hazmat.primitives.kdf.pbkdf2', 'cryptography.hazmat.backends.openssl', 'cv2', 'pandas', 'pandas._libs.tslibs.base', 'matplotlib.backends.backend_qt5agg', 'PyQt5.sip', 'sklearn.utils._typedefs', 'sklearn.neighbors._typedefs', 'sklearn.neighbors._quad_tree', 'sklearn.tree', 'sklearn.tree._utils'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='区域检测器',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='区域检测器',
)
app = BUNDLE(
    coll,
    name='区域检测器.app',
    icon=None,
    bundle_identifier=None,
)
