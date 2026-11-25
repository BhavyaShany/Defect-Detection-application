import base64
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

from backend.config import settings
from backend.services import geometry_store as GS


class TestGeometryStore(unittest.TestCase):
    def test_roundtrip_with_and_without_faces_and_traversal_guard(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(settings, "DATA_DIR", d):
            sid = str(uuid.uuid4())
            pos = np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32)
            sc = np.linspace(0, 1, 100).astype(np.float32)
            out = (np.arange(100) % 10 == 0)
            faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
            self.assertGreater(GS.save_geometry(sid, pos, sc, out, faces), 0)
            p = GS.load_geometry_payload(sid)
            got = np.frombuffer(base64.b64decode(p["positions_b64"]), dtype="<f4").reshape(-1, 3)
            np.testing.assert_array_equal(got, pos)
            self.assertEqual(p["face_count"], 2)
            self.assertIsNone(p["defect_mask_b64"])
            np.testing.assert_array_equal(
                np.frombuffer(base64.b64decode(p["faces_b64"]), dtype="<u4").reshape(-1, 3), faces)

            sid2 = str(uuid.uuid4())
            GS.save_geometry(sid2, pos, sc, out, None)
            self.assertIsNone(GS.load_geometry_payload(sid2)["faces_b64"])

            self.assertIsNone(GS.load_geometry_payload("../../etc/passwd"))
            self.assertIsNone(GS.load_geometry_payload(str(uuid.uuid4())))  # never stored
            GS.delete_geometry(sid)
            self.assertIsNone(GS.load_geometry_payload(sid))
            self.assertEqual([f.name for f in Path(d, "geometry").glob("*.tmp")], [])  # no temp leaks


if __name__ == "__main__":
    unittest.main()
