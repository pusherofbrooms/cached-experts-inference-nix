"""Header-only contract checks; no tensor payloads are needed or validated."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from runtime.prepare_model import validate_metal_mtp


class MetalSchema(unittest.TestCase):
    def test_all_tensor_contracts(self):
        schema = json.loads(Path('runtime/metal_mtp_schema.json').read_text())
        def string(s):
            b = s.encode()
            return struct.pack('<Q', len(b)) + b
        def header(items):
            b = b'GGUF' + struct.pack('<IQQ', 3, 31, 2)
            for k, v in [('general.architecture', 'qwen4exp-mtp'),
                         ('strata.mtp.expert_format', 'q2_0')]:
                b += string(k) + struct.pack('<I', 8) + string(v)
            for name, (shape, kind) in items:
                b += string(name) + struct.pack('<I', len(shape))
                b += struct.pack('<' + 'Q' * len(shape), *shape)
                b += struct.pack('<IQ', kind, 0)
            return b
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'header.gguf'
            p.write_bytes(header(list(schema.items())))
            validate_metal_mtp(p)  # Success with no payload proves metadata-only.
            for name, (shape, kind) in schema.items():
                for replacement in [('mtp.wrong', (shape, kind)),
                                    (name, ([*shape[:-1], shape[-1] + 1], kind)),
                                    (name, ([2**63] * len(shape), kind)),
                                    (name, (shape, kind + 1))]:
                    with self.subTest(name=name, replacement=replacement):
                        items = [(n, v) if n != name else replacement for n, v in schema.items()]
                        p.write_bytes(header(items))
                        with self.assertRaises(ValueError):
                            validate_metal_mtp(p)
            p.write_bytes(b'GGUF' + struct.pack('<IQQQ', 3, 31, 1, 2**40))
            with self.assertRaises(ValueError):
                validate_metal_mtp(p)
