"""The secret scanner: it must catch keys and private files, and pass a clean copy of this folder."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scan_for_secrets as scan  # noqa: E402


class TestSecretScanner(unittest.TestCase):
    def test_this_package_is_clean(self):
        self.assertEqual(scan.scan(ROOT), [])

    def test_catches_keys_tokens_and_private_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # fake values, assembled at run time so this file itself never looks like it holds a key
            fake_one = "P" + "K" + "ABCDEFGHIJ1234567890"
            fake_two = "aB3dE5fG7hJ9kL1m" + "N3pQ5rS7tU9vW1xY3zA5bC7d"
            fake_three = "https://discord.com/api/" + "webhooks/123456789/" + "AbCdEfGhIjKlMnOpQrStUvWx"
            (d / "bot.py").write_text(f'API_KEY = "{fake_one}"\nSECRET_KEY = "{fake_two}"\nhook = "{fake_three}"\n')
            (d / ".env").write_text("ALPACA_API_KEY=x\n")
            (d / "sparticus_v6_3.log").write_text("hi\n")
            (d / "sparticus_journal.csv").write_text("symbol\n")
            (d / "lab_data").mkdir()
            found = "\n".join(scan.scan(d))
        for what in ("Alpaca API key id", "secret value", "Discord webhook", ".env", "log file", "trade journal",
                     "lab_data"):
            self.assertIn(what, found)
        self.assertNotIn("1234567890\"", found)     # values are masked in the output

    def test_template_placeholders_are_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / ".env.example").write_text((ROOT / ".env.example").read_text())
            (d / "bot.py").write_text('API_KEY = os.getenv("ALPACA_API_KEY")\nSECRET_KEY = None\n')
            self.assertEqual(scan.scan(d), [])

    def test_names_and_paths_are_not_mistaken_for_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "app.py").write_text('secrets = secrets_from_ssm(client)\n'
                                      'SECRET_KEY_PARAM: /my-bot/alpaca_secret_key\n')
            (d / "cacert.pem").write_text("-----BEGIN CERTIFICATE-----\nabc\n-----END CERTIFICATE-----\n")
            self.assertEqual(scan.scan(d), [])

    def test_aws_build_output_and_terraform_state_are_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / ".aws-sam").mkdir()
            (d / "terraform.tfstate").write_text("{}")
            found = "\n".join(scan.scan(d))
        self.assertIn(".aws-sam", found)
        self.assertIn("Terraform state", found)


if __name__ == "__main__":
    unittest.main()
