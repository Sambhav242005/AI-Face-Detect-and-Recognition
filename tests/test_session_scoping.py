import asyncio
import json
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

    def test_session_list_empty_after_cleanup(self):
        result = run(main.list_sessions())
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["sessions"]), 0)

    def test_session_list_after_creation(self):
        s1 = run(main.new_session(model_name="edgeface_xxs"))
        s2 = run(main.new_session(model_name=main.DEFAULT_MODEL_NAME))

        result = run(main.list_sessions())
        self.assertEqual(len(result["sessions"]), 2)
        ids = {s["session_id"] for s in result["sessions"]}
        self.assertIn(s1["session_id"], ids)
        self.assertIn(s2["session_id"], ids)

    def test_session_load_returns_face_count(self):
        created = run(main.new_session(model_name=main.DEFAULT_MODEL_NAME))
        embedding = basis_embedding(0)
        run(main.query_face({
            "session_id": created["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "embedding": embedding,
            "allow_new_identity": True,
        }))

        loaded = run(main.load_session(
            created["session_id"],
            model_name=main.DEFAULT_MODEL_NAME,
        ))
        self.assertEqual(loaded["face_count"], 1)
        self.assertEqual(loaded["unique_identities"], 1)

    def test_delete_face_removes_identity(self):
        created = run(main.new_session(model_name=main.DEFAULT_MODEL_NAME))
        embedding = basis_embedding(0)
        run(main.query_face({
            "session_id": created["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "embedding": embedding,
            "allow_new_identity": True,
        }))

        # Verify reid=1 exists
        loaded = run(main.load_session(created["session_id"]))
        self.assertEqual(loaded["face_count"], 1)

        # Delete it
        result = run(main.delete_face(created["session_id"], "1"))
        self.assertEqual(result["status"], "success")

        # Verify it's gone
        loaded = run(main.load_session(created["session_id"]))
        self.assertEqual(loaded["face_count"], 0)

    def test_delete_nonexistent_face_returns_404(self):
        created = run(main.new_session(model_name=main.DEFAULT_MODEL_NAME))
        with self.assertRaises(HTTPException) as ctx:
            run(main.delete_face(created["session_id"], "999"))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_merge_faces_reassigns_embeddings(self):
        created = run(main.new_session(model_name=main.DEFAULT_MODEL_NAME))
        e1 = basis_embedding(0)
        e2 = basis_embedding(1)

        r1 = run(main.query_face({
            "session_id": created["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "embedding": e1,
            "allow_new_identity": True,
        }))
        self.assertEqual(r1["reid"], 1)

        r2 = run(main.query_face({
            "session_id": created["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "embedding": e2,
            "allow_new_identity": True,
        }))
        self.assertEqual(r2["reid"], 2)

        # Merge reid 2 into reid 1
        merge_result = run(main.merge_faces({
            "session_id": created["session_id"],
            "merge_from": 2,
            "merge_to": 1,
        }))
        self.assertEqual(merge_result["status"], "success")

        # Both embeddings should now be under reid 1
        loaded = run(main.load_session(created["session_id"]))
        self.assertEqual(loaded["face_count"], 2)
        self.assertEqual(loaded["unique_identities"], 1)

    def test_merge_self_is_rejected(self):
        created = run(main.new_session(model_name=main.DEFAULT_MODEL_NAME))
        with self.assertRaises(HTTPException) as ctx:
            run(main.merge_faces({
                "session_id": created["session_id"],
                "merge_from": 1,
                "merge_to": 1,
            }))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_name_validation_rejects_invalid_chars(self):
        created = run(main.new_session(model_name=main.DEFAULT_MODEL_NAME))
        with self.assertRaises(HTTPException) as ctx:
            run(main.update_face({
                "session_id": created["session_id"],
                "model_name": main.DEFAULT_MODEL_NAME,
                "reid": 1,
                "name": "<script>alert('xss')</script>",
            }))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_name_validation_accepts_valid_name(self):
        created = run(main.new_session(model_name=main.DEFAULT_MODEL_NAME))
        embedding = basis_embedding(0)
        run(main.query_face({
            "session_id": created["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "embedding": embedding,
            "allow_new_identity": True,
        }))

        result = run(main.update_face({
            "session_id": created["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "reid": 1,
            "name": "Alice Smith",
        }))
        self.assertEqual(result["status"], "success")

    def test_tracker_override_with_expansion(self):
        created = run(main.new_session(model_name=main.DEFAULT_MODEL_NAME))

        # Add identity 1
        e1 = basis_embedding(0)
        r1 = run(main.query_face({
            "session_id": created["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "embedding": e1,
            "allow_new_identity": True,
        }))
        self.assertEqual(r1["reid"], 1)

        # Add identity 2
        e2 = basis_embedding(100)
        r2 = run(main.query_face({
            "session_id": created["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "embedding": e2,
            "allow_new_identity": True,
        }))
        self.assertEqual(r2["reid"], 2)

        # Query identity 2's embedding with known_reid=1 (old tracker) — this tests override logic
        # Since e2 is very different from e1 (>0.4), but e2 matches reid 2 well
        result = run(main.query_face({
            "session_id": created["session_id"],
            "model_name": main.DEFAULT_MODEL_NAME,
            "embedding": e2,
            "track_id": 99,
            "known_reid": 1,
            "allow_new_identity": False,
            "allow_profile_expansion": True,
        }))
        # Should match identity 2
        self.assertEqual(result["reid"], 2)


if __name__ == "__main__":
    unittest.main()
