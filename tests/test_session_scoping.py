import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import main


def run(coro):
    return asyncio.run(coro)


def basis_embedding(index: int):
    values = [0.0] * 512
    values[index] = 1.0
    return values


class SessionScopingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_session_dir = main.SESSION_DIR
        main.SESSION_DIR = Path(self.temp_dir.name)
        main.clear_session_cache()

    def tearDown(self):
        main.clear_session_cache()
        main.SESSION_DIR = self.original_session_dir
        self.temp_dir.cleanup()

    def test_face_queries_are_scoped_to_requested_session(self):
        first = run(main.new_session(model_name=main.DEFAULT_MODEL_NAME))
        second = run(main.new_session(model_name=main.DEFAULT_MODEL_NAME))

        embedding = basis_embedding(0)
        created = run(main.query_face({
            "session_id": first["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "embedding": embedding,
            "allow_new_identity": True,
        }))
        self.assertEqual(created["reid"], 1)

        run(main.update_face({
            "session_id": first["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "reid": 1,
            "name": "Alice",
        }))

        isolated = run(main.query_face({
            "session_id": second["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "embedding": embedding,
            "allow_new_identity": False,
        }))
        self.assertIsNone(isolated["reid"])
        self.assertEqual(isolated["name"], "Unknown")

        matched = run(main.query_face({
            "session_id": first["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "embedding": embedding,
            "allow_new_identity": False,
        }))
        self.assertEqual(matched["reid"], 1)
        self.assertEqual(matched["name"], "Alice")

    def test_model_mismatch_is_rejected(self):
        created = run(main.new_session(model_name="edgeface_xxs"))
        main.clear_session_cache()

        with self.assertRaises(HTTPException) as ctx:
            run(main.load_session(
                created["session_id"],
                model_name=main.DEFAULT_MODEL_NAME,
            ))

        self.assertEqual(ctx.exception.status_code, 409)

    def test_face_query_requires_session_id(self):
        with self.assertRaises(HTTPException) as ctx:
            run(main.query_face({
                "model_name": main.DEFAULT_MODEL_NAME,
                "embedding": basis_embedding(0),
                "allow_new_identity": True,
            }))

        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
